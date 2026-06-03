from oxidize_python.core.paged.paged import Scheduler, default_scheduler_config


def test_scheduler_add_and_step() -> None:
    s = Scheduler(default_scheduler_config())
    r = s.add_request([1, 2, 3], 10)
    assert r.id == 1
    scheduled = s.step()
    assert scheduled and scheduled[0].id == 1
