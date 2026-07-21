/* test_mesh.c — distributed mesh stub tests. */
#include <criterion/criterion.h>
#include "oxidize/mesh.h"
#include <string.h>

Test(mesh, init_leader)
{
    OcMesh mesh;
    OcMeshConfig cfg = OC_MESH_CONFIG_DEFAULT;
    cfg.is_leader = true;
    cr_assert_eq(oc_mesh_init(&mesh, &cfg), OC_OK);
    cr_assert(mesh.initialized);
    cr_assert_eq(oc_mesh_peer_count(&mesh), 1);
    const OcPeer *p = oc_mesh_get_peer(&mesh, 0);
    cr_assert_not_null(p);
    cr_assert_eq(p->role, OC_PEER_LEADER);
    cr_assert(p->online);
    oc_mesh_free(&mesh);
    cr_assert(!mesh.initialized);
}

Test(mesh, init_follower)
{
    OcMesh mesh;
    OcMeshConfig cfg = OC_MESH_CONFIG_DEFAULT;
    cfg.is_leader = false;
    cr_assert_eq(oc_mesh_init(&mesh, &cfg), OC_OK);
    const OcPeer *p = oc_mesh_get_peer(&mesh, 0);
    cr_assert_eq(p->role, OC_PEER_FOLLOWER);
    oc_mesh_free(&mesh);
}

Test(mesh, connect_peer)
{
    OcMesh mesh;
    OcMeshConfig cfg = OC_MESH_CONFIG_DEFAULT;
    cfg.is_leader = true;
    cr_assert_eq(oc_mesh_init(&mesh, &cfg), OC_OK);
    cr_assert_eq(oc_mesh_connect(&mesh, "10.0.0.2:5000"), OC_OK);
    cr_assert_eq(oc_mesh_peer_count(&mesh), 2);
    const OcPeer *p = oc_mesh_get_peer(&mesh, 1);
    cr_assert_not_null(p);
    cr_assert_str_eq(p->addr, "10.0.0.2:5000");
    oc_mesh_free(&mesh);
}

Test(mesh, shard_layout_single_node)
{
    OcMesh mesh;
    OcMeshConfig cfg = OC_MESH_CONFIG_DEFAULT;
    cfg.tensor_parallel = 1;
    cr_assert_eq(oc_mesh_init(&mesh, &cfg), OC_OK);
    OcShardLayout shard;
    cr_assert_eq(oc_mesh_shard_for_layer(&mesh, 0, 1000, &shard), OC_OK);
    cr_assert_eq(shard.rank, 0);
    cr_assert_eq(shard.world_size, 1);
    cr_assert_eq(shard.shard_start, 0);
    cr_assert_eq(shard.shard_end, 1000);
    oc_mesh_free(&mesh);
}

Test(mesh, shard_layout_tp4)
{
    OcMesh mesh;
    OcMeshConfig cfg = OC_MESH_CONFIG_DEFAULT;
    cfg.tensor_parallel = 4;
    cr_assert_eq(oc_mesh_init(&mesh, &cfg), OC_OK);
    OcShardLayout shard;
    cr_assert_eq(oc_mesh_shard_for_layer(&mesh, 0, 1000, &shard), OC_OK);
    cr_assert_eq(shard.world_size, 4);
    cr_assert_eq(shard.rank, 0);
    cr_assert_eq(shard.shard_end - shard.shard_start, 250);
    oc_mesh_free(&mesh);
}

Test(mesh, broadcast_and_allreduce)
{
    OcMesh mesh;
    OcMeshConfig cfg = OC_MESH_CONFIG_DEFAULT;
    cr_assert_eq(oc_mesh_init(&mesh, &cfg), OC_OK);
    /* Single-node: these are no-ops. */
    float data[] = {1.0f, 2.0f, 3.0f};
    cr_assert_eq(oc_mesh_broadcast(&mesh, data, sizeof(data), 0), OC_OK);
    cr_assert_eq(oc_mesh_allreduce(&mesh, data, 3), OC_OK);
    cr_assert_float_eq(data[0], 1.0f, 1e-6f);
    oc_mesh_free(&mesh);
}
