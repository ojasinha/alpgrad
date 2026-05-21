# alpgrad

alpgrad ("alp" meaning "little" in Hindi/Sanskrit) is a port of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) in pure C.
- Dependency free. Just standard C.
- Lightweight and fast.

## Project Structure

- `alpgrad/` contains all the header and code files.
- `data/` contains data for examples.
- `tests/` contains sanity checks and other operations tests.
- `examples/` contains example demonstrations.

## Installation

You can download the `alpgrad` directory, drop it in your project and start using it directly.

## Examples

Clone the repository, cd into it and then run

```bash
make
```
to compile the examples and tests into a newly created `builds` folder.

You can then run the tests and examples from the root directory like so (example_2 needs to be run like this to not mess up the file path because it's using `data/make_moons.csv`)
```bash
./builds/test
./builds/example_1
./builds/example_2
```
