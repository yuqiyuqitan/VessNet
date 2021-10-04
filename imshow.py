# helper function to show image(s), channels first
def imshow(raw, ground_truth=None, prediction=None):
  rows = 1
  if ground_truth is not None:
    rows += 1
  if prediction is not None:
    rows += 1
  cols = raw.shape[0] if len(raw.shape) > 3 else 1
  fig, axes = plt.subplots(rows, cols, figsize=(10, 4), sharex=True, sharey=True, squeeze=False)
  if len(raw.shape) == 3:
    axes[0][0].imshow(raw.transpose(1, 2, 0))
  else:
    for i, im in enumerate(raw):
      axes[0][i].imshow(im.transpose(1, 2, 0))
  row = 1
  if ground_truth is not None:
    if len(ground_truth.shape) == 3:
      axes[row][0].imshow(ground_truth[0])
    else:
      for i, gt in enumerate(ground_truth):
        axes[row][i].imshow(gt[0])
    row += 1
  if prediction is not None:
    if len(prediction.shape) == 3:
      axes[row][0].imshow(prediction[0])
    else:
      for i, gt in enumerate(prediction):
        axes[row][i].imshow(gt[0])
  plt.show()