# Data usage

The dataset for this project is supplied through the R.O.A.D. Barbados Historic Handwriting Challenge on Zindi. It is not part of this repository.

## What must remain private

- `Train.csv`, `Test.csv`, and `SampleSubmission.csv`
- all competition images and archives
- validation targets, prediction caches, and error-analysis tables
- generated submissions and other row-level derivatives

The repository `.gitignore` excludes these files and the public notebooks have their saved outputs removed. Obtain the data directly from the competition page and follow the current Zindi rules and data terms.

## Expected local layout

```text
resources/
  Train.csv
  Test.csv
  SampleSubmission.csv
  images/<ID>.jpg
```

Code assumes that image filenames match the `ID` column. Do not replace the empty `resources/.gitkeep` file with actual data in a commit.

## Validation policy

The curriculum uses a deterministic 400-image development holdout and excludes 14 known unusable IDs. Stage 2 continues from the best Stage 1 checkpoint. KenLM is fitted only on training labels; its fusion weight is selected on a deterministic 100-image calibration subset and evaluated on the remaining 300 images.

The reported score is a local proxy. It should not be described as an official Zindi leaderboard score.

