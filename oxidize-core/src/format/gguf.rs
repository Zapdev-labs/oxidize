use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(target_os = "linux")]
use libc;
#[cfg(unix)]
use memmap2::Advice;
use memmap2::Mmap;
use thiserror::Error;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

/// Read `MemAvailable` from `/proc/meminfo` (Linux only).
/// Returns `None` on any parse failure; callers treat that as "unlimited" to be safe.
#[cfg(target_os = "linux")]
pub fn linux_mem_available_bytes() -> Option<u64> {
    let data = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in data.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: BTreeMap<String, GgufMetadataValue>,
    pub tensor_infos: Vec<GgufTensorInfo>,
    pub alignment: u64,
    pub data_section_start: u64,
}

#[derive(Debug, Clone)]
pub struct MappedGgufFile {
    mmaps: Vec<Arc<Mmap>>,
    parsed: GgufFile,
}

impl PartialEq for MappedGgufFile {
    fn eq(&self, other: &Self) -> bool {
        self.parsed == other.parsed
    }
}

impl MappedGgufFile {
    pub fn parsed(&self) -> &GgufFile {
        &self.parsed
    }

    /// Returns the bytes of the first (or only) shard. For single-file GGUFs
    /// this is the whole model; for split GGUFs use `tensor_mmap` / `tensor_bytes`
    /// when you need a specific tensor's backing bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.mmaps[0]
    }

    /// Total byte length across all shards (useful for size reporting).
    pub fn total_bytes_len(&self) -> u64 {
        self.mmaps.iter().map(|m| m.len() as u64).sum()
    }

    /// Arc to the first (or only) shard's mmap.
    pub fn mmap(&self) -> Arc<Mmap> {
        Arc::clone(&self.mmaps[0])
    }

    /// Arc to the shard mmap that contains this tensor.
    pub fn tensor_mmap(&self, info: &GgufTensorInfo) -> Arc<Mmap> {
        Arc::clone(&self.mmaps[info.mmap_index])
    }

    /// Byte slice for this tensor's data within its shard.
    pub fn tensor_bytes(&self, info: &GgufTensorInfo, size: usize) -> &[u8] {
        let mmap = &self.mmaps[info.mmap_index];
        let start = info.absolute_offset as usize;
        &mmap[start..start + size]
    }

    #[cfg(test)]
    pub fn from_parsed_for_test(parsed: GgufFile) -> Self {
        Self {
            mmaps: vec![std::sync::Arc::new(
                memmap2::MmapOptions::new()
                    .len(1)
                    .map_anon()
                    .unwrap()
                    .make_read_only()
                    .unwrap(),
            )],
            parsed,
        }
    }

    pub fn advise_random_access(&self) -> std::io::Result<()> {
        #[cfg(unix)]
        for mmap in &self.mmaps {
            mmap.advise(Advice::Random)?;
        }
        Ok(())
    }

    pub fn advise_will_need(&self) -> std::io::Result<()> {
        #[cfg(unix)]
        for mmap in &self.mmaps {
            mmap.advise(Advice::WillNeed)?;
        }
        Ok(())
    }

    /// Enable THP only when the model fits in RAM with ≥2× headroom.
    /// On file-backed MAP_PRIVATE mmaps, MADV_HUGEPAGE causes khugepaged to
    /// create anonymous 2 MiB copies of every file page, consuming as much RAM
    /// as the model size in anonymous memory — defeating the purpose of mmap for
    /// large models.  Skip it when the model would exhaust available RAM.
    #[cfg(target_os = "linux")]
    pub fn advise_huge_pages(&self) -> std::io::Result<()> {
        let model_bytes = self.total_bytes_len();
        let available = linux_mem_available_bytes().unwrap_or(0);
        // Only enable THP when model is <50% of available RAM (2× headroom).
        if model_bytes > 0 && available > 0 && model_bytes * 2 <= available {
            const MADV_COLLAPSE: libc::c_int = 25;
            for mmap in &self.mmaps {
                mmap.advise(Advice::HugePage)?;
                // MADV_HUGEPAGE only hints khugepaged, which in practice never
                // collapses read-only file pages while decode is running — the
                // model stays in 4 KB pages and every token's full weight sweep
                // pays a TLB walk per 64 cache lines (~600K walks/token for a
                // 2.5 GB model). MADV_COLLAPSE (kernel >= 6.1) collapses the
                // page-cache folios synchronously at load. Best effort: older
                // kernels return EINVAL and we keep the khugepaged hint.
                let bytes: &[u8] = mmap;
                unsafe {
                    libc::madvise(
                        bytes.as_ptr() as *mut libc::c_void,
                        bytes.len(),
                        MADV_COLLAPSE,
                    );
                }
            }
            Ok(())
        } else {
            Ok(())
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn advise_huge_pages(&self) -> std::io::Result<()> {
        Ok(())
    }

    /// Touch every page sequentially to fault them into the page cache.
    pub fn prefault_pages(&self) -> u8 {
        let bytes = self.bytes();
        let mut checksum = 0_u8;
        for offset in (0..bytes.len()).step_by(4096) {
            checksum ^= crate::bytes::read_volatile_byte(bytes, offset);
        }
        if let Some(last) = bytes.last() {
            checksum ^= *last;
        }
        checksum
    }

    /// Lock pages into physical RAM and fault every page in parallel.
    ///
    /// On Linux with `CAP_IPC_LOCK`:
    /// 1. Raise `RLIMIT_MEMLOCK` to unlimited.
    /// 2. Check `MemAvailable` — only call `mlock` when model fits with headroom
    ///    (model_bytes < available_bytes * 70%).  Plain `mlock` faults every page
    ///    immediately; without headroom it races the model loader for physical RAM
    ///    and triggers the OOM killer.
    /// 3. When mlock is skipped, fall back to `madvise(WILLNEED)` which queues
    ///    async readahead without reserving physical pages.
    /// 4. Parallel read_volatile sweep to saturate all memory channels.
    ///
    /// Returns `(mlocked, checksum, duration_ms)`.
    pub fn prefault_pages_locked(&self, threads: usize) -> (bool, u8, u64) {
        let t0 = std::time::Instant::now();
        let bytes = self.bytes();
        let mlocked = false;

        #[cfg(target_os = "linux")]
        {
            let mut mlocked = mlocked;
            // Raise RLIMIT_MEMLOCK (requires CAP_IPC_LOCK or root).
            let unlimited = libc::rlimit {
                rlim_cur: libc::RLIM_INFINITY,
                rlim_max: libc::RLIM_INFINITY,
            };
            // SAFETY: valid rlimit struct.
            unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &unlimited) };

            // Only mlock when the model fits with ≥30% headroom so the model loader
            // and KV-cache allocator have room to breathe.
            let available = linux_mem_available_bytes().unwrap_or(u64::MAX);
            let model_bytes = bytes.len() as u64;
            let safe_to_lock = model_bytes < available.saturating_mul(7) / 10;

            if safe_to_lock {
                // Plain mlock faults ALL pages synchronously and pins them so the
                // kernel cannot evict them under memory pressure.
                // SAFETY: valid pointer + length for the mmap region we own.
                let rc = unsafe { libc::mlock(bytes.as_ptr() as *const libc::c_void, bytes.len()) };
                mlocked = rc == 0;
            }

            if !mlocked {
                // Headroom too tight for mlock — fall back to async readahead only.
                // Pages land in page-cache and will be warm for inference but can be
                // evicted under pressure.
                for mmap in &self.mmaps {
                    let _ = mmap.advise(Advice::WillNeed);
                }
            }
        }

        // Parallel read_volatile sweep — saturates all memory channels and ensures
        // every page is faulted in even when mlock wasn't used.
        use rayon::prelude::*;
        let page_size = 4096_usize;
        let n_threads = threads.max(1);

        let offsets: Vec<usize> = (0..bytes.len()).step_by(page_size).collect();
        let ptr = bytes.as_ptr() as usize;

        let checksum: u8 = offsets
            .par_chunks(offsets.len().div_ceil(n_threads).max(1))
            .map(|chunk| {
                let mut c = 0_u8;
                for &off in chunk {
                    // SAFETY: off is in-bounds by construction; ptr is valid for the
                    // lifetime of `self` which outlives this closure.
                    c ^= unsafe { std::ptr::read_volatile((ptr + off) as *const u8) };
                }
                c
            })
            .reduce(|| 0_u8, |a, b| a ^ b);

        let ms = t0.elapsed().as_millis() as u64;
        (mlocked, checksum, ms)
    }

    pub fn mapped_tensor_infos(&self) -> Vec<GgufTensorInfo> {
        self.parsed.mapped_tensor_infos()
    }
}

impl GgufFile {
    pub fn architecture(&self) -> Option<&str> {
        if let Some(GgufMetadataValue::String(value)) = self.metadata.get("general.architecture") {
            return Some(value.as_str());
        }
        detect_architecture_from_metadata_keys(&self.metadata)
    }

    pub fn mapped_tensor_infos(&self) -> Vec<GgufTensorInfo> {
        let architecture = self.architecture().unwrap_or("llama");
        self.tensor_infos
            .iter()
            .map(|tensor| GgufTensorInfo {
                name: map_tensor_name(architecture, &tensor.name),
                dimensions: tensor.dimensions.clone(),
                ggml_type: tensor.ggml_type,
                relative_offset: tensor.relative_offset,
                absolute_offset: tensor.absolute_offset,
                mmap_index: tensor.mmap_index,
            })
            .collect()
    }

    pub fn quantization_type(&self) -> Option<GgufQuantizationType> {
        let file_type = metadata_as_u32(self.metadata.get("general.file_type")?)?;
        Some(GgufQuantizationType::from_llama_ftype(file_type))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub relative_offset: u64,
    pub absolute_offset: u64,
    /// Index into `MappedGgufFile::mmaps` that holds this tensor's data.
    /// Always 0 for single-file GGUFs; identifies the shard for split files.
    pub mmap_index: usize,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufQuantizationType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K_S,
    Q3_K_M,
    Q3_K_L,
    Q4_K_S,
    Q4_K_M,
    Q5_K_S,
    Q5_K_M,
    Q6_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    IQ1_M,
    I8,
    I16,
    I32,
    I64,
    F64,
    BF16,
    NVFP4,
    Unknown(u32),
}

impl GgufQuantizationType {
    /// `general.file_type` in GGUF stores `enum llama_ftype` from llama.cpp (`include/llama.h`).
    pub fn from_llama_ftype(ftype: u32) -> Self {
        match ftype {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            7 => Self::Q8_0,
            8 => Self::Q5_0,
            9 => Self::Q5_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K_S,
            12 => Self::Q3_K_M,
            13 => Self::Q3_K_L,
            14 => Self::Q4_K_S,
            15 => Self::Q4_K_M,
            16 => Self::Q5_K_S,
            17 => Self::Q5_K_M,
            18 => Self::Q6_K,
            1024 => Self::Unknown(ftype),
            other => Self::Unknown(other),
        }
    }

    /// Per-tensor `ggml_type` field in GGUF tensor info (`enum ggml_type` in ggml).
    pub fn from_ggml_type(ggml_type: u32) -> Self {
        match ggml_type {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2_K,
            11 => Self::Q3_K_S, // ggml Q3_K
            12 => Self::Q4_K_M, // ggml Q4_K (S/M distinction is metadata-level, not type-level)
            13 => Self::Q5_K_M, // ggml Q5_K
            14 => Self::Q6_K,   // ggml Q6_K
            15 => Self::Q4_K_M, // ggml Q8_K — no Q8_K enum variant; closest supported type
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            31 => Self::Q4_0,
            32 => Self::Q8_0,
            33 => Self::IQ2_XXS,
            34 => Self::IQ2_XS,
            35 => Self::IQ3_XXS,
            40 => Self::NVFP4,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(GgufMetadataArray),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufMetadataArray {
    pub element_type: GgufMetadataType,
    pub values: Vec<GgufMetadataValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufMetadataType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufMetadataType {
    type Error = GgufParseError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(GgufParseError::UnknownMetadataType(value)),
        }
    }
}

#[derive(Debug, Error)]
pub enum GgufParseError {
    #[error("invalid gguf magic")]
    InvalidMagic,
    #[error("unsupported gguf version: {0}")]
    UnsupportedVersion(u32),
    #[error("unexpected end of file")]
    UnexpectedEof,
    #[error("invalid utf8 string: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("unknown metadata type: {0}")]
    UnknownMetadataType(u32),
    #[error("invalid alignment: {0}")]
    InvalidAlignment(u64),
    #[error("integer overflow while parsing")]
    IntegerOverflow,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl PartialEq for GgufParseError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::InvalidMagic, Self::InvalidMagic) => true,
            (Self::UnsupportedVersion(a), Self::UnsupportedVersion(b)) => a == b,
            (Self::UnexpectedEof, Self::UnexpectedEof) => true,
            (Self::InvalidUtf8(a), Self::InvalidUtf8(b)) => a.utf8_error() == b.utf8_error(),
            (Self::UnknownMetadataType(a), Self::UnknownMetadataType(b)) => a == b,
            (Self::InvalidAlignment(a), Self::InvalidAlignment(b)) => a == b,
            (Self::IntegerOverflow, Self::IntegerOverflow) => true,
            (Self::Io(a), Self::Io(b)) => a.kind() == b.kind(),
            _ => false,
        }
    }
}

pub fn load_mapped_gguf<P: AsRef<Path>>(path: P) -> Result<MappedGgufFile, GgufParseError> {
    let path = path.as_ref();
    if let Some(shards) = collect_split_shards(path) {
        return load_mapped_gguf_shards(&shards);
    }
    let file = File::open(path)?;
    // SAFETY: The returned mapping is read-only and we keep it alive for as long as
    // parsed metadata is exposed from MappedGgufFile.
    // SAFETY: GGUF files are opened read-only and not modified while mapped.
    let mmap = unsafe { crate::bytes::map_readonly(&file)? };
    // Tell the kernel we'll read the whole file front-to-back and that it should
    // start prefetching it immediately.  This queues async readahead so pages are
    // warm by the time inference begins — critical for multi-hundred-GB models.
    #[cfg(unix)]
    {
        let _ = mmap.advise(Advice::Sequential);
        let _ = mmap.advise(Advice::WillNeed);
    }
    let parsed = parse_gguf(&mmap)?;
    Ok(MappedGgufFile {
        mmaps: vec![Arc::new(mmap)],
        parsed,
    })
}

/// If `path` matches the pattern `<base>-NNNNN-of-MMMMM.gguf` and all sibling
/// shards exist on disk, returns them in order (shard 1 … MMMMM). Returns
/// `None` for regular (non-split) GGUF files.
fn collect_split_shards(path: &Path) -> Option<Vec<PathBuf>> {
    let filename = path.file_name()?.to_str()?;
    let stem = filename.strip_suffix(".gguf")?;
    let (prefix_with_no, total_str) = stem.rsplit_once("-of-")?;
    let total: usize = total_str.parse().ok().filter(|&n| n >= 2)?;
    let base = prefix_with_no.rsplit_once('-')?.0;
    let dir = path.parent().unwrap_or(Path::new("."));
    let shards: Vec<PathBuf> = (1..=total)
        .map(|i| dir.join(format!("{base}-{i:05}-of-{total:05}.gguf")))
        .collect();
    if shards.iter().all(|p| p.exists()) {
        Some(shards)
    } else {
        None
    }
}

/// Load a set of split-GGUF shard paths (in order) into a single `MappedGgufFile`.
/// Shard 0 provides all metadata; subsequent shards contribute only their tensors.
fn load_mapped_gguf_shards(shards: &[PathBuf]) -> Result<MappedGgufFile, GgufParseError> {
    debug_assert!(!shards.is_empty());

    let open_mmap = |path: &PathBuf| -> Result<Mmap, GgufParseError> {
        let file = File::open(path)?;
        // SAFETY: GGUF files are opened read-only and not modified while mapped.
        let mmap = unsafe { crate::bytes::map_readonly(&file)? };
        #[cfg(unix)]
        {
            let _ = mmap.advise(Advice::Sequential);
            let _ = mmap.advise(Advice::WillNeed);
        }
        Ok(mmap)
    };

    let mmap0 = open_mmap(&shards[0])?;
    let mut parsed0 = parse_gguf(&mmap0)?;
    // Stamp shard-0 tensors with mmap_index = 0.
    for t in &mut parsed0.tensor_infos {
        t.mmap_index = 0;
    }

    let mut mmaps: Vec<Arc<Mmap>> = vec![Arc::new(mmap0)];

    for (idx, shard_path) in shards[1..].iter().enumerate() {
        let mmap = open_mmap(shard_path)?;
        let mut parsed = parse_gguf(&mmap)?;
        let mmap_index = idx + 1;
        for t in &mut parsed.tensor_infos {
            t.mmap_index = mmap_index;
        }
        parsed0.tensor_infos.extend(parsed.tensor_infos);
        parsed0.tensor_count += parsed.tensor_count;
        mmaps.push(Arc::new(mmap));
    }

    Ok(MappedGgufFile {
        mmaps,
        parsed: parsed0,
    })
}

pub fn parse_gguf(bytes: &[u8]) -> Result<GgufFile, GgufParseError> {
    let mut reader = ByteReader::new(bytes);

    let magic = reader.read_exact(4)?;
    if magic != GGUF_MAGIC {
        return Err(GgufParseError::InvalidMagic);
    }

    let version = reader.read_u32()?;
    if version != 2 && version != 3 {
        return Err(GgufParseError::UnsupportedVersion(version));
    }

    let tensor_count = reader.read_u64()?;
    let metadata_count = reader.read_u64()?;

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = reader.read_string()?;
        let value_type = GgufMetadataType::try_from(reader.read_u32()?)?;
        let value = reader.read_value_of_type(value_type)?;
        metadata.insert(key, value);
    }

    let mut tensor_infos = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = reader.read_string()?;
        let n_dimensions = reader.read_u32()?;
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(reader.read_u64()?);
        }
        let ggml_type = reader.read_u32()?;
        let relative_offset = reader.read_u64()?;
        tensor_infos.push(GgufTensorInfo {
            name,
            dimensions,
            ggml_type,
            relative_offset,
            absolute_offset: 0,
            mmap_index: 0,
        });
    }

    let alignment = metadata
        .get("general.alignment")
        .map_or(Ok(DEFAULT_ALIGNMENT), alignment_from_metadata)?;
    if !alignment.is_power_of_two() {
        return Err(GgufParseError::InvalidAlignment(alignment));
    }

    let data_section_start = align_up(reader.position() as u64, alignment)?;
    if data_section_start > bytes.len() as u64 {
        return Err(GgufParseError::UnexpectedEof);
    }

    for tensor in &mut tensor_infos {
        tensor.absolute_offset = data_section_start
            .checked_add(tensor.relative_offset)
            .ok_or(GgufParseError::IntegerOverflow)?;
        if tensor.absolute_offset > bytes.len() as u64 {
            return Err(GgufParseError::UnexpectedEof);
        }
    }

    Ok(GgufFile {
        version,
        tensor_count,
        metadata,
        tensor_infos,
        alignment,
        data_section_start,
    })
}

fn alignment_from_metadata(value: &GgufMetadataValue) -> Result<u64, GgufParseError> {
    match value {
        GgufMetadataValue::Uint8(v) => Ok((*v).into()),
        GgufMetadataValue::Uint16(v) => Ok((*v).into()),
        GgufMetadataValue::Uint32(v) => Ok((*v).into()),
        GgufMetadataValue::Uint64(v) => Ok(*v),
        GgufMetadataValue::Int8(v) if *v > 0 => Ok(*v as u64),
        GgufMetadataValue::Int16(v) if *v > 0 => Ok(*v as u64),
        GgufMetadataValue::Int32(v) if *v > 0 => Ok(*v as u64),
        GgufMetadataValue::Int64(v) if *v > 0 => Ok(*v as u64),
        _ => Err(GgufParseError::InvalidAlignment(0)),
    }
}

fn metadata_as_u32(value: &GgufMetadataValue) -> Option<u32> {
    match value {
        GgufMetadataValue::Uint8(v) => Some((*v).into()),
        GgufMetadataValue::Uint16(v) => Some((*v).into()),
        GgufMetadataValue::Uint32(v) => Some(*v),
        GgufMetadataValue::Uint64(v) => (*v).try_into().ok(),
        GgufMetadataValue::Int8(v) if *v >= 0 => Some((*v as u8).into()),
        GgufMetadataValue::Int16(v) if *v >= 0 => Some((*v as u16).into()),
        GgufMetadataValue::Int32(v) if *v >= 0 => (*v).try_into().ok(),
        GgufMetadataValue::Int64(v) if *v >= 0 => (*v).try_into().ok(),
        _ => None,
    }
}

fn detect_architecture_from_metadata_keys(
    metadata: &BTreeMap<String, GgufMetadataValue>,
) -> Option<&str> {
    for key in metadata.keys() {
        let Some((namespace, _)) = key.split_once('.') else {
            continue;
        };
        let architecture = match namespace {
            "llama" | "mistral" | "mixtral" | "qwen" | "qwen2" | "qwen2moe" | "qwen35"
            | "deepseek" | "deepseek2" | "deepseek_v2" | "deepseek_v3" | "deepseek_moe"
            | "gemma" | "phi" | "falcon" | "gpt2" | "gptj" | "gptneox" | "dflash"
            | "dflash-draft" | "glm-dsa" | "glm_dsa" | "glm_moe_dsa" => Some(namespace),
            _ => None,
        };
        if architecture.is_some() {
            return architecture;
        }
    }
    None
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufParseError> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or(GgufParseError::IntegerOverflow)
}

fn map_tensor_name(architecture: &str, name: &str) -> String {
    let architecture = architecture.to_ascii_lowercase();
    let mapped = match architecture.as_str() {
        "llama" | "mistral" | "mixtral" | "qwen" | "qwen2" | "qwen2moe" | "qwen35" | "deepseek"
        | "deepseek2" | "deepseek_v2" | "deepseek_v3" | "deepseek_moe" | "gemma" | "phi"
        | "glm-dsa" | "glm_dsa" | "glm_moe_dsa" => map_hf_decoder_name(name),
        "falcon" => map_falcon_name(name),
        "gpt2" => map_gpt2_name(name),
        "gptj" => map_gptj_name(name),
        "gptneox" => map_gpt_neox_name(name),
        _ => None,
    };
    mapped.unwrap_or_else(|| name.to_owned())
}

fn map_hf_decoder_name(name: &str) -> Option<String> {
    match name {
        "model.embed_tokens.weight" => Some("tok_embeddings.weight".to_owned()),
        "lm_head.weight" => Some("output.weight".to_owned()),
        "model.norm.weight" => Some("norm.weight".to_owned()),
        _ => {
            let (layer, suffix) = name.strip_prefix("model.layers.")?.split_once('.')?;
            if let Some(rest) = suffix.strip_prefix("block_sparse_moe.experts.") {
                let (expert, expert_weight) = rest.split_once('.')?;
                let mapped_expert_weight = match expert_weight {
                    "w1.weight" => "ffn_gate",
                    "w2.weight" => "ffn_down",
                    "w3.weight" => "ffn_up",
                    _ => return None,
                };
                return Some(format!(
                    "blk.{layer}.{mapped_expert_weight}.{expert}.weight"
                ));
            }
            if let Some(rest) = suffix.strip_prefix("mlp.experts.") {
                let (expert, expert_weight) = rest.split_once('.')?;
                let mapped_expert_weight = match expert_weight {
                    "gate_proj.weight" => "ffn_gate",
                    "up_proj.weight" => "ffn_up",
                    "down_proj.weight" => "ffn_down",
                    _ => return None,
                };
                return Some(format!(
                    "blk.{layer}.{mapped_expert_weight}.{expert}.weight"
                ));
            }
            let mapped_suffix = match suffix {
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.q_a_proj.weight" => "attn_q_a.weight",
                "self_attn.q_a_layernorm.weight" => "attn_q_a_norm.weight",
                "self_attn.q_b_proj.weight" => "attn_q_b.weight",
                "self_attn.kv_a_proj_with_mqa.weight" => "attn_kv_a_mqa.weight",
                "self_attn.kv_a_layernorm.weight" => "attn_kv_a_norm.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                "mlp.gate.weight" => "ffn_gate_inp.weight",
                "mlp.shared_expert.gate_proj.weight" => "ffn_gate_shexp.weight",
                "mlp.shared_expert.up_proj.weight" => "ffn_up_shexp.weight",
                "mlp.shared_expert.down_proj.weight" => "ffn_down_shexp.weight",
                "mlp.shared_expert_gate.weight" => "ffn_gate_inp_shexp.weight",
                "block_sparse_moe.gate.weight" => "ffn_gate_inp.weight",
                _ => return None,
            };
            Some(format!("blk.{layer}.{mapped_suffix}"))
        }
    }
}

fn map_falcon_name(name: &str) -> Option<String> {
    match name {
        "transformer.word_embeddings.weight" => Some("tok_embeddings.weight".to_owned()),
        "lm_head.weight" => Some("output.weight".to_owned()),
        "transformer.ln_f.weight" => Some("norm.weight".to_owned()),
        _ => None,
    }
}

fn map_gpt2_name(name: &str) -> Option<String> {
    match name {
        "transformer.wte.weight" => Some("tok_embeddings.weight".to_owned()),
        "lm_head.weight" => Some("output.weight".to_owned()),
        "transformer.ln_f.weight" => Some("norm.weight".to_owned()),
        _ => None,
    }
}

fn map_gptj_name(name: &str) -> Option<String> {
    match name {
        "transformer.wte.weight" => Some("tok_embeddings.weight".to_owned()),
        "lm_head.weight" => Some("output.weight".to_owned()),
        "transformer.ln_f.weight" => Some("norm.weight".to_owned()),
        _ => None,
    }
}

fn map_gpt_neox_name(name: &str) -> Option<String> {
    match name {
        "gpt_neox.embed_in.weight" => Some("tok_embeddings.weight".to_owned()),
        "embed_out.weight" | "lm_head.weight" => Some("output.weight".to_owned()),
        "gpt_neox.final_layer_norm.weight" => Some("norm.weight".to_owned()),
        _ => None,
    }
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn position(&self) -> usize {
        self.cursor
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], GgufParseError> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or(GgufParseError::IntegerOverflow)?;
        if end > self.bytes.len() {
            return Err(GgufParseError::UnexpectedEof);
        }
        let out = &self.bytes[self.cursor..end];
        self.cursor = end;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8, GgufParseError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufParseError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufParseError> {
        let mut buf = [0_u8; 2];
        buf.copy_from_slice(self.read_exact(2)?);
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Result<i16, GgufParseError> {
        let mut buf = [0_u8; 2];
        buf.copy_from_slice(self.read_exact(2)?);
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> Result<u32, GgufParseError> {
        let mut buf = [0_u8; 4];
        buf.copy_from_slice(self.read_exact(4)?);
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32, GgufParseError> {
        let mut buf = [0_u8; 4];
        buf.copy_from_slice(self.read_exact(4)?);
        Ok(i32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64, GgufParseError> {
        let mut buf = [0_u8; 8];
        buf.copy_from_slice(self.read_exact(8)?);
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64, GgufParseError> {
        let mut buf = [0_u8; 8];
        buf.copy_from_slice(self.read_exact(8)?);
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32, GgufParseError> {
        let mut buf = [0_u8; 4];
        buf.copy_from_slice(self.read_exact(4)?);
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64, GgufParseError> {
        let mut buf = [0_u8; 8];
        buf.copy_from_slice(self.read_exact(8)?);
        Ok(f64::from_le_bytes(buf))
    }

    fn read_bool(&mut self) -> Result<bool, GgufParseError> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> Result<String, GgufParseError> {
        let len = self.read_u64()?;
        let len: usize = len
            .try_into()
            .map_err(|_| GgufParseError::IntegerOverflow)?;
        let bytes = self.read_exact(len)?;
        Ok(String::from_utf8(bytes.to_vec())?)
    }

    fn read_value_of_type(
        &mut self,
        value_type: GgufMetadataType,
    ) -> Result<GgufMetadataValue, GgufParseError> {
        match value_type {
            GgufMetadataType::Uint8 => Ok(GgufMetadataValue::Uint8(self.read_u8()?)),
            GgufMetadataType::Int8 => Ok(GgufMetadataValue::Int8(self.read_i8()?)),
            GgufMetadataType::Uint16 => Ok(GgufMetadataValue::Uint16(self.read_u16()?)),
            GgufMetadataType::Int16 => Ok(GgufMetadataValue::Int16(self.read_i16()?)),
            GgufMetadataType::Uint32 => Ok(GgufMetadataValue::Uint32(self.read_u32()?)),
            GgufMetadataType::Int32 => Ok(GgufMetadataValue::Int32(self.read_i32()?)),
            GgufMetadataType::Float32 => Ok(GgufMetadataValue::Float32(self.read_f32()?)),
            GgufMetadataType::Bool => Ok(GgufMetadataValue::Bool(self.read_bool()?)),
            GgufMetadataType::String => Ok(GgufMetadataValue::String(self.read_string()?)),
            GgufMetadataType::Array => {
                let element_type = GgufMetadataType::try_from(self.read_u32()?)?;
                let len = self.read_u64()?;
                let mut values = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    values.push(self.read_value_of_type(element_type)?);
                }
                Ok(GgufMetadataValue::Array(GgufMetadataArray {
                    element_type,
                    values,
                }))
            }
            GgufMetadataType::Uint64 => Ok(GgufMetadataValue::Uint64(self.read_u64()?)),
            GgufMetadataType::Int64 => Ok(GgufMetadataValue::Int64(self.read_i64()?)),
            GgufMetadataType::Float64 => Ok(GgufMetadataValue::Float64(self.read_f64()?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    fn fixture_bytes(name: &str) -> Vec<u8> {
        fs::read(fixture_path(name)).expect("fixture file exists")
    }

    #[test]
    fn parses_v3_header_tensor_info_and_alignment() {
        let bytes = fixture_bytes("valid-v3.gguf");

        let parsed = parse_gguf(&bytes).expect("parse succeeds");

        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.tensor_count, 1);
        assert_eq!(parsed.alignment, 64);
        assert_eq!(parsed.data_section_start, 128);
        assert_eq!(parsed.tensor_infos.len(), 1);
        assert_eq!(parsed.tensor_infos[0].name, "tok_embeddings.weight");
        assert_eq!(parsed.tensor_infos[0].dimensions, vec![32000, 4096]);
        assert_eq!(parsed.tensor_infos[0].absolute_offset, 128);
    }

    #[test]
    fn rejects_invalid_magic() {
        let bytes = fixture_bytes("invalid-magic.gguf");

        let err = parse_gguf(&bytes).expect_err("must reject magic");
        assert_eq!(err, GgufParseError::InvalidMagic);
    }

    #[test]
    fn rejects_unsupported_version() {
        let bytes = fixture_bytes("unsupported-version.gguf");

        let err = parse_gguf(&bytes).expect_err("must reject version");
        assert_eq!(err, GgufParseError::UnsupportedVersion(1));
    }

    #[test]
    fn rejects_invalid_alignment_value() {
        let bytes = fixture_bytes("invalid-alignment.gguf");

        let err = parse_gguf(&bytes).expect_err("must reject invalid alignment");
        assert_eq!(err, GgufParseError::InvalidAlignment(3));
    }

    #[test]
    fn loads_gguf_via_memory_map() {
        let path = fixture_path("valid-v3.gguf");
        let bytes = fixture_bytes("valid-v3.gguf");

        let mapped = load_mapped_gguf(&path).expect("mapped load succeeds");

        assert_eq!(mapped.parsed().version, 3);
        assert_eq!(mapped.parsed().tensor_count, 1);
        assert_eq!(mapped.parsed().alignment, 64);
        assert_eq!(mapped.parsed().tensor_infos[0].absolute_offset, 128);
        assert_eq!(mapped.bytes(), bytes.as_slice());
    }

    #[test]
    fn mapped_load_returns_io_error_for_missing_file() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock is after epoch")
            .as_nanos();
        let path = env::temp_dir().join(format!("oxidize-core-missing-{unique}.gguf"));

        let err = load_mapped_gguf(&path).expect_err("missing file should error");
        assert!(matches!(err, GgufParseError::Io(_)));
    }

    #[test]
    fn maps_llama_hf_tensor_names_to_internal_format() {
        let file = GgufFile {
            version: 3,
            tensor_count: 3,
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                GgufMetadataValue::String("llama".to_owned()),
            )]),
            tensor_infos: vec![
                tensor_info("model.embed_tokens.weight"),
                tensor_info("model.layers.3.self_attn.q_proj.weight"),
                tensor_info("model.layers.3.mlp.down_proj.weight"),
            ],
            alignment: 32,
            data_section_start: 0,
        };

        let mapped = file.mapped_tensor_infos();
        assert_eq!(mapped[0].name, "tok_embeddings.weight");
        assert_eq!(mapped[1].name, "blk.3.attn_q.weight");
        assert_eq!(mapped[2].name, "blk.3.ffn_down.weight");
    }

    #[test]
    fn maps_qwen_tensor_names_to_internal_format() {
        let qwen2 = GgufFile {
            version: 3,
            tensor_count: 2,
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                GgufMetadataValue::String("qwen2".to_owned()),
            )]),
            tensor_infos: vec![
                tensor_info("model.embed_tokens.weight"),
                tensor_info("model.layers.1.self_attn.k_proj.weight"),
            ],
            alignment: 32,
            data_section_start: 0,
        };
        let qwen2moe = GgufFile {
            version: 3,
            tensor_count: 1,
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                GgufMetadataValue::String("qwen2moe".to_owned()),
            )]),
            tensor_infos: vec![tensor_info(
                "model.layers.4.block_sparse_moe.experts.2.w1.weight",
            )],
            alignment: 32,
            data_section_start: 0,
        };

        let qwen2_mapped = qwen2.mapped_tensor_infos();
        let qwen2moe_mapped = qwen2moe.mapped_tensor_infos();
        assert_eq!(qwen2_mapped[0].name, "tok_embeddings.weight");
        assert_eq!(qwen2_mapped[1].name, "blk.1.attn_k.weight");
        assert_eq!(qwen2moe_mapped[0].name, "blk.4.ffn_gate.2.weight");
    }

    #[test]
    fn maps_gemma_tensor_names_to_internal_format() {
        let file = GgufFile {
            version: 3,
            tensor_count: 2,
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                GgufMetadataValue::String("gemma".to_owned()),
            )]),
            tensor_infos: vec![
                tensor_info("model.embed_tokens.weight"),
                tensor_info("model.layers.6.self_attn.o_proj.weight"),
            ],
            alignment: 32,
            data_section_start: 0,
        };

        let mapped = file.mapped_tensor_infos();
        assert_eq!(mapped[0].name, "tok_embeddings.weight");
        assert_eq!(mapped[1].name, "blk.6.attn_output.weight");
    }

    #[test]
    fn maps_phi_tensor_names_to_internal_format() {
        let file = GgufFile {
            version: 3,
            tensor_count: 3,
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                GgufMetadataValue::String("phi".to_owned()),
            )]),
            tensor_infos: vec![
                tensor_info("model.embed_tokens.weight"),
                tensor_info("model.layers.4.self_attn.v_proj.weight"),
                tensor_info("model.layers.4.mlp.gate_proj.weight"),
            ],
            alignment: 32,
            data_section_start: 0,
        };

        let mapped = file.mapped_tensor_infos();
        assert_eq!(mapped[0].name, "tok_embeddings.weight");
        assert_eq!(mapped[1].name, "blk.4.attn_v.weight");
        assert_eq!(mapped[2].name, "blk.4.ffn_gate.weight");
    }

    #[test]
    fn maps_falcon_gpt2_gptj_and_gptneox_embedding_names() {
        let falcon = GgufFile {
            version: 3,
            tensor_count: 1,
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                GgufMetadataValue::String("falcon".to_owned()),
            )]),
            tensor_infos: vec![tensor_info("transformer.word_embeddings.weight")],
            alignment: 32,
            data_section_start: 0,
        };
        let gpt2 = GgufFile {
            version: 3,
            tensor_count: 1,
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                GgufMetadataValue::String("gpt2".to_owned()),
            )]),
            tensor_infos: vec![tensor_info("transformer.wte.weight")],
            alignment: 32,
            data_section_start: 0,
        };
        let gptj = GgufFile {
            version: 3,
            tensor_count: 1,
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                GgufMetadataValue::String("gptj".to_owned()),
            )]),
            tensor_infos: vec![tensor_info("transformer.wte.weight")],
            alignment: 32,
            data_section_start: 0,
        };
        let gptneox = GgufFile {
            version: 3,
            tensor_count: 1,
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                GgufMetadataValue::String("gptneox".to_owned()),
            )]),
            tensor_infos: vec![tensor_info("gpt_neox.embed_in.weight")],
            alignment: 32,
            data_section_start: 0,
        };

        assert_eq!(
            falcon.mapped_tensor_infos()[0].name,
            "tok_embeddings.weight"
        );
        assert_eq!(gpt2.mapped_tensor_infos()[0].name, "tok_embeddings.weight");
        assert_eq!(gptj.mapped_tensor_infos()[0].name, "tok_embeddings.weight");
        assert_eq!(
            gptneox.mapped_tensor_infos()[0].name,
            "tok_embeddings.weight"
        );
    }

    #[test]
    fn maps_gpt_output_and_norm_names() {
        let gpt2 = GgufFile {
            version: 3,
            tensor_count: 2,
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                GgufMetadataValue::String("gpt2".to_owned()),
            )]),
            tensor_infos: vec![
                tensor_info("lm_head.weight"),
                tensor_info("transformer.ln_f.weight"),
            ],
            alignment: 32,
            data_section_start: 0,
        };
        let gptneox = GgufFile {
            version: 3,
            tensor_count: 2,
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                GgufMetadataValue::String("gptneox".to_owned()),
            )]),
            tensor_infos: vec![
                tensor_info("embed_out.weight"),
                tensor_info("gpt_neox.final_layer_norm.weight"),
            ],
            alignment: 32,
            data_section_start: 0,
        };

        let gpt2_mapped = gpt2.mapped_tensor_infos();
        let gptneox_mapped = gptneox.mapped_tensor_infos();
        assert_eq!(gpt2_mapped[0].name, "output.weight");
        assert_eq!(gpt2_mapped[1].name, "norm.weight");
        assert_eq!(gptneox_mapped[0].name, "output.weight");
        assert_eq!(gptneox_mapped[1].name, "norm.weight");
    }

    #[test]
    fn keeps_original_name_when_no_mapping_rule_matches() {
        let file = GgufFile {
            version: 3,
            tensor_count: 1,
            metadata: BTreeMap::new(),
            tensor_infos: vec![tensor_info("custom.tensor.weight")],
            alignment: 32,
            data_section_start: 0,
        };

        let mapped = file.mapped_tensor_infos();
        assert_eq!(mapped[0].name, "custom.tensor.weight");
    }

    #[test]
    fn architecture_prefers_general_architecture_metadata_when_present() {
        let file = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: BTreeMap::from([
                (
                    "general.architecture".to_owned(),
                    GgufMetadataValue::String("qwen2".to_owned()),
                ),
                (
                    "llama.context_length".to_owned(),
                    GgufMetadataValue::Uint32(4096),
                ),
            ]),
            tensor_infos: Vec::new(),
            alignment: 32,
            data_section_start: 0,
        };

        assert_eq!(file.architecture(), Some("qwen2"));
    }

    #[test]
    fn architecture_detects_namespace_when_general_architecture_is_missing() {
        let file = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: BTreeMap::from([(
                "qwen2.context_length".to_owned(),
                GgufMetadataValue::Uint32(32768),
            )]),
            tensor_infos: Vec::new(),
            alignment: 32,
            data_section_start: 0,
        };

        assert_eq!(file.architecture(), Some("qwen2"));
    }

    #[test]
    fn architecture_detects_dflash_namespace_when_general_architecture_is_missing() {
        let file = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: BTreeMap::from([(
                "dflash.embedding_length".to_owned(),
                GgufMetadataValue::Uint32(2048),
            )]),
            tensor_infos: Vec::new(),
            alignment: 32,
            data_section_start: 0,
        };

        assert_eq!(file.architecture(), Some("dflash"));
    }

    #[test]
    fn architecture_detects_deepseek_namespace_when_general_architecture_is_missing() {
        let file = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: BTreeMap::from([(
                "deepseek2.expert_count".to_owned(),
                GgufMetadataValue::Uint32(384),
            )]),
            tensor_infos: Vec::new(),
            alignment: 32,
            data_section_start: 0,
        };

        assert_eq!(file.architecture(), Some("deepseek2"));
    }

    #[test]
    fn architecture_returns_none_for_unknown_namespaces() {
        let file = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: BTreeMap::from([(
                "tokenizer.ggml.model".to_owned(),
                GgufMetadataValue::String("bpe".to_owned()),
            )]),
            tensor_infos: Vec::new(),
            alignment: 32,
            data_section_start: 0,
        };

        assert_eq!(file.architecture(), None);
    }

    #[test]
    fn maps_mixtral_moe_tensor_names_to_internal_format() {
        let file = GgufFile {
            version: 3,
            tensor_count: 4,
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                GgufMetadataValue::String("mixtral".to_owned()),
            )]),
            tensor_infos: vec![
                tensor_info("model.layers.2.block_sparse_moe.gate.weight"),
                tensor_info("model.layers.2.block_sparse_moe.experts.3.w1.weight"),
                tensor_info("model.layers.2.block_sparse_moe.experts.3.w2.weight"),
                tensor_info("model.layers.2.block_sparse_moe.experts.3.w3.weight"),
            ],
            alignment: 32,
            data_section_start: 0,
        };

        let mapped = file.mapped_tensor_infos();
        assert_eq!(mapped[0].name, "blk.2.ffn_gate_inp.weight");
        assert_eq!(mapped[1].name, "blk.2.ffn_gate.3.weight");
        assert_eq!(mapped[2].name, "blk.2.ffn_down.3.weight");
        assert_eq!(mapped[3].name, "blk.2.ffn_up.3.weight");
    }

    #[test]
    fn maps_deepseek_moe_and_shared_expert_tensor_names_to_internal_format() {
        let file = GgufFile {
            version: 3,
            tensor_count: 7,
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                GgufMetadataValue::String("deepseek2".to_owned()),
            )]),
            tensor_infos: vec![
                tensor_info("model.layers.1.self_attn.q_a_proj.weight"),
                tensor_info("model.layers.1.self_attn.kv_a_proj_with_mqa.weight"),
                tensor_info("model.layers.1.mlp.gate.weight"),
                tensor_info("model.layers.1.mlp.experts.42.gate_proj.weight"),
                tensor_info("model.layers.1.mlp.shared_expert.gate_proj.weight"),
                tensor_info("model.layers.1.mlp.shared_expert.up_proj.weight"),
                tensor_info("model.layers.1.mlp.shared_expert_gate.weight"),
            ],
            alignment: 32,
            data_section_start: 0,
        };

        let mapped = file.mapped_tensor_infos();
        assert_eq!(mapped[0].name, "blk.1.attn_q_a.weight");
        assert_eq!(mapped[1].name, "blk.1.attn_kv_a_mqa.weight");
        assert_eq!(mapped[2].name, "blk.1.ffn_gate_inp.weight");
        assert_eq!(mapped[3].name, "blk.1.ffn_gate.42.weight");
        assert_eq!(mapped[4].name, "blk.1.ffn_gate_shexp.weight");
        assert_eq!(mapped[5].name, "blk.1.ffn_up_shexp.weight");
        assert_eq!(mapped[6].name, "blk.1.ffn_gate_inp_shexp.weight");
    }

    #[test]
    fn detects_known_quantization_types() {
        let file = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: BTreeMap::from([(
                "general.file_type".to_owned(),
                GgufMetadataValue::Uint32(15),
            )]),
            tensor_infos: Vec::new(),
            alignment: 32,
            data_section_start: 0,
        };
        assert_eq!(file.quantization_type(), Some(GgufQuantizationType::Q4_K_M));

        let file = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: BTreeMap::from([(
                "general.file_type".to_owned(),
                GgufMetadataValue::Uint8(17),
            )]),
            tensor_infos: Vec::new(),
            alignment: 32,
            data_section_start: 0,
        };
        assert_eq!(file.quantization_type(), Some(GgufQuantizationType::Q5_K_M));
        assert_eq!(
            GgufQuantizationType::from_ggml_type(40),
            GgufQuantizationType::NVFP4
        );
    }

    #[test]
    fn detects_unknown_quantization_type_value() {
        let file = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: BTreeMap::from([(
                "general.file_type".to_owned(),
                GgufMetadataValue::Uint32(999),
            )]),
            tensor_infos: Vec::new(),
            alignment: 32,
            data_section_start: 0,
        };
        assert_eq!(
            file.quantization_type(),
            Some(GgufQuantizationType::Unknown(999))
        );
    }

    #[test]
    fn returns_none_when_quantization_type_missing_or_invalid() {
        let missing = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: BTreeMap::new(),
            tensor_infos: Vec::new(),
            alignment: 32,
            data_section_start: 0,
        };
        assert_eq!(missing.quantization_type(), None);

        let invalid = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: BTreeMap::from([(
                "general.file_type".to_owned(),
                GgufMetadataValue::String("Q4_K_M".to_owned()),
            )]),
            tensor_infos: Vec::new(),
            alignment: 32,
            data_section_start: 0,
        };
        assert_eq!(invalid.quantization_type(), None);
    }

    #[test]
    fn parser_handles_seeded_bytes_without_panicking() {
        fn seeded_bytes(seed: u64, len: usize) -> Vec<u8> {
            let mut state = seed;
            let mut out = Vec::with_capacity(len);
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                out.push((state & 0xFF) as u8);
            }
            out
        }

        for len in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let bytes = seeded_bytes(0x5EED_1234_5678_9ABC, len);
            let _ = parse_gguf(&bytes);
        }
    }

    fn tensor_info(name: &str) -> GgufTensorInfo {
        GgufTensorInfo {
            name: name.to_owned(),
            dimensions: vec![1],
            ggml_type: 0,
            relative_offset: 0,
            absolute_offset: 0,
            mmap_index: 0,
        }
    }

    // ── from_ggml_type correctness ────────────────────────────────────────────

    #[test]
    fn from_ggml_type_k_quants_map_correctly() {
        // These mappings must match the ggml spec (ggml.h enum ggml_type).
        // Real ggml.h mapping: Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15.
        // Oxidize uses Q4_K_M for type 12 (S/M is metadata-level in ggml, not type-level).
        let cases = [
            (10, GgufQuantizationType::Q2_K),
            (11, GgufQuantizationType::Q3_K_S),
            (12, GgufQuantizationType::Q4_K_M),
            (13, GgufQuantizationType::Q5_K_M),
            (14, GgufQuantizationType::Q6_K),
            (15, GgufQuantizationType::Q4_K_M),
            (30, GgufQuantizationType::BF16),
        ];
        for (id, expected) in cases {
            assert_eq!(
                GgufQuantizationType::from_ggml_type(id),
                expected,
                "ggml_type {id} should map to {expected:?}"
            );
        }
    }

    #[test]
    fn from_ggml_type_basic_types_unchanged() {
        assert_eq!(
            GgufQuantizationType::from_ggml_type(0),
            GgufQuantizationType::F32
        );
        assert_eq!(
            GgufQuantizationType::from_ggml_type(1),
            GgufQuantizationType::F16
        );
        assert_eq!(
            GgufQuantizationType::from_ggml_type(8),
            GgufQuantizationType::Q8_0
        );
    }

    // ── prefault_pages_locked ─────────────────────────────────────────────────

    /// Build a minimal GGUF with a small payload and verify that
    /// `prefault_pages_locked` returns and produces the same checksum as
    /// `prefault_pages` across several thread counts.
    fn make_padded_gguf() -> (tempfile::NamedTempFile, Vec<u8>) {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut f = NamedTempFile::new().expect("tmpfile");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3_u32.to_le_bytes()); // version
        data.extend_from_slice(&0_u64.to_le_bytes()); // n_tensors
        data.extend_from_slice(&0_u64.to_le_bytes()); // n_kv
        // Pad out to several pages so parallel chunking exercises all branches.
        data.resize(4096 * 8, 0xAB_u8);
        f.write_all(&data).unwrap();
        f.flush().unwrap();
        (f, data)
    }

    #[test]
    fn prefault_pages_locked_consistent_across_thread_counts() {
        let (f, _) = make_padded_gguf();
        let mapped = load_mapped_gguf(f.path()).expect("open tmpfile");

        // All thread counts must produce identical checksums since they
        // XOR the same set of page offsets.
        let (_mlocked, base, _) = mapped.prefault_pages_locked(1);
        for threads in [2, 4, 8] {
            let (_mlocked, checksum, ms) = mapped.prefault_pages_locked(threads);
            assert_eq!(
                checksum, base,
                "checksum mismatch with {threads} threads (took {ms}ms)"
            );
        }
    }

    #[test]
    fn prefault_pages_locked_completes_quickly() {
        let (f, _) = make_padded_gguf();
        let mapped = load_mapped_gguf(f.path()).expect("open tmpfile");
        let (_, _, ms) = mapped.prefault_pages_locked(4);
        assert!(ms < 5000, "prefault should be fast (took {ms}ms)");
    }
}
