import click
from project_heart import scripts

# - - - - - - - - - - - - - - - - - - - - -
# Main click group
@click.group()
def main():
    pass

# - - - - - - - - - - - - - - - - - - - - -
# run command -> click sub group
@main.group(
    short_help="Executes a given command. Ask 'run --help' to learn more.",
    help="Executes one of the commands listed below.")
def run():
    pass


# extract features data
@run.command(short_help="hello", help="hello")
@click.option("--json_file", '-i', 
    type=click.Path(dir_okay=False, file_okay=True), 
    help="")
def extract_geometrics(**kargs):
  return scripts.extract_geometrics(**kargs)


# extract features data
@run.command(short_help="hello", help="hello")
@click.option("--json_file", '-i', 
    type=click.Path(dir_okay=False, file_okay=True), 
    help="")
def compute_fibers(**kargs):
  return scripts.compute_fibers(**kargs)