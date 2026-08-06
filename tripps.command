#!/usr/bin/env bash
# Double-click me.
#
# macOS runs a .command file in Terminal when you double-click it — a plain .sh opens in an
# editor instead, which is why this file exists next to run.sh. All the work is in run.sh;
# this only finds the folder it lives in (Terminal starts in your home directory, not here)
# and keeps the window open if something goes wrong, so the error is still readable.

cd "$(dirname "$0")" || exit 1

./run.sh "$@"      # a double-click passes nothing; from a terminal you can still add args
status=$?

if [ $status -ne 0 ] && [ $status -ne 130 ]; then   # 130 is ctrl-c, which is how you stop it
  echo
  echo "tripps exited with status $status. The error is above."
  echo "Press any key to close this window."
  read -r -n 1 -s
fi

exit $status
