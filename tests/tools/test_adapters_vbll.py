from bllarse.tools.adapters import config_to_argv


def test_config_to_argv_negates_listed_bools():
    argv = config_to_argv({"return_ood": False, "px": False, "epochs": 3})
    assert "--no-return-ood" in argv
    assert "--px" not in argv
    assert "--no-px" not in argv
    assert argv[argv.index("--epochs") + 1] == "3"


def test_config_to_argv_keeps_true_bools_as_bare_flags():
    argv = config_to_argv({"return_ood": True})
    assert argv == ["--return-ood"]
