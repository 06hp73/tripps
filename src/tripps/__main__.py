"""Let `python -m tripps` work, so nothing depends on a generated console script.

pip writes the interpreter's absolute path into the shebang of `bin/tripps` at install time.
That is fine for a virtualenv that stays put, and fatal for a bundle: moving the interpreter
into tripps.app left every console script pointing at the machine it was built on
("bad interpreter: No such file or directory"). Running the module instead means the
interpreter being used is the one invoking us, wherever it now lives.
"""

from __future__ import annotations

from .cli import main

raise SystemExit(main())
