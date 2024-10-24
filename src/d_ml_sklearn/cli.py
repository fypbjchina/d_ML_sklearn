"""Console script for d_ml_sklearn."""
import d_ml_sklearn

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for d_ml_sklearn."""
    console.print("Replace this message by putting your code into "
               "d_ml_sklearn.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
