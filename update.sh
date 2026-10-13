#!/bin/sh
# Update Strata without starting the model (#475): what ./setup.sh does before a start - the engine (a new ready-made
# one when this version needs it, or compiled again when its source changed), the Python packages, each installed
# model's settings and draft subset. The model files are not touched. Start the model later with ./setup.sh. Options
# are passed on to setup.py. Nothing is fetched from the network here: new code comes with a new Docker image build.
cd "$(dirname "$0")" || exit 1
exec sh ./setup.sh --update "$@"
