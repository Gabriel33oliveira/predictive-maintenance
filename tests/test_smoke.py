import predmaint
from predmaint import cli


def test_version_is_non_empty_string():
    assert isinstance(predmaint.__version__, str)
    assert predmaint.__version__


def test_cli_version_prints_version(capsys):
    try:
        code = cli.main(["--version"])
    except SystemExit as exc:
        code = exc.code
    assert code in (0, None)
    assert predmaint.__version__ in capsys.readouterr().out


def test_cli_without_args_returns_zero():
    assert cli.main([]) == 0
