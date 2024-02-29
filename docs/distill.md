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

A configuration file is a [YAML](https://yaml.org/) file that specify how to create a student model. This repository comes with two valid configuration files that you can peruse:

* [data/distill_examples/lambda_00/config.yaml](../data/distill_examples/lambda_00/config.yaml)
* [data/distill_examples/lambda_01/config.yaml](../data/distill_examples/lambda_01/config.yaml)

I recommend that you use the `distiller_ui` program to create configuration files rather than writing them yourself. Inside the program, you can see what the fields are and what they mean.

## What `distill` Outputs

Inside the configuration file, you specify a directory where the student models should be saved to in the `prefix` field. After `distill` is done with its job, the output directory will look like this:

```
+ <prefix-specified-in-config-file>
  + body_morpher
  + face_morpher
  + character_model
  - config.yaml
```

Here:

* `config.yaml` is a copy of the configuration file that you wrote. 
* The `character_model` directory contains a trained student model that can be used with `character_model_manual_poser.md`, `character_model_ifacialmocap_puppeteer.md`, and `character_model_mediapipe_puppeteer.md`. 
* `body_morpher` is a scratch directory that was used to save intermediate results during the training of a part of the student model.
* `face_morpher` is a scratch directory that was used to save intermediate results during the training of another part of the student model.

You only need what is inside the `character_model` directory. As a resulit, you can delete other files after the `character_model` directory has been filled. You can move the directory out to somewhere and rename it as long as the contents inside are not modified.

## The Training Process Is Interruptible

Invoking `distill` on a configuration will start a rather long process of training a student model. On a machine with an A6000 GPU, it takes about 30 hours to complete. As a result, it might take several days on machines with less powerful GPUs.

The training process is robust and interruptible. You can stop it any time by closing the shell window or by typing `Ctrl+C`. Intermediate results are periodically saved in the scratch directories, ready to be picked up at a later time when you are ready to train the student model again. To resume the process, just invoke `distill` again with the same configuration file that you started with, and the process will take care of itself.