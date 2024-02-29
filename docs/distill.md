# `distill`

This program trains a student model given a configuration file, a $512 \times 512$ RGBA character image, and a mask of facial organs.

## Invoking the Program

Make sure you have (1) created a Python environment and (2) downloaded model files as instruction in the [main README file](../README.md).

### Instruction for Linux/OSX Users

1. Open a shell.
2. `cd` to the repository's directory.
   ```
   cd SOMEWHERE/talking-head-anime-4-demo
   ```
3. Run the program.
   ```
   bin/run src/tha4/app/distill.py <config-file>
   ```
   where `<config-file>` is a configuration file for creating a student model. More on this later.

### Instruction for Windows Users

1. Open a shell.
2. `cd` to the repository's directory.
   ```
   cd SOMEWHERE\talking-head-anime-4-demo
   ```
3. Run the program.
   ```
   bin\run.bat src\tha4\app\full_manual_poser.py <config-file>
   ```   
   where `<config-file>` is a configuration file for creating a student model. More on this later.

## Configuration File

