from inference.cli import served_name


def test_hub_repo_keeps_its_name():
    assert served_name('andnetdeboer/some_model') == 'andnetdeboer/some_model'


def test_local_folder_drops_the_machine_path():
    assert served_name('/data/users/someone/models/BlueCupDit_ep_40/step-030000') == 'BlueCupDit_ep_40/step-030000'


def test_revision_is_kept():
    assert served_name('andnetdeboer/some_model', 'step-005000') == 'andnetdeboer/some_model@step-005000'
