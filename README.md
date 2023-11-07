# 10th solution private 0.61 : TNT/VIT/CAIT ensemble + tricks
to be updated ….

there are two part:

part one: TNT/VIT/CAIT ensemble model (public/private LB 0.71/0.71)
https://www.kaggle.com/c/bms-molecular-translation/discussion/243766 (this post)
part two: OCR + MolBuilder + object detection CNN (public/private LB 0.62/0.61)
https://www.kaggle.com/c/bms-molecular-translation/discussion/243809
TNT/VIT/CAIT ensemble model (public/private LB 0.71/0.71)

we train 5 models : 3x patched based VIT (patch size 16, image scale = 0.8,0.9,0.1), 1x TNT with image size 384 and 1x CAIT with image size 240. For details, please refer to https://www.kaggle.com/c/bms-molecular-translation/discussion/231190
(local CV 0.78)

we do 5 iterations of image perturbation. if an image has gives an invalid inchi (verification by rdkit), we perturb the image and try inference again.
( local CV 0.76)

finally, we apply rdkit normalisation.
(public/private LB 0.73/0.74, local CV 0.72)

The ensemble is our top score submission. We also have submissions for each of the individual models. We then sort all submission cvs by LB score. To create the final submission:

1. We start off will predictions from the top score submission csv.
2. For a current prediction has invalid inchi, it will be replaced from the corresponding one from the next higher score submission csv if ( and only if ) that corresponding one is valid
3. The process is repeated for all csvs.
Final LB score is (public/private LB 0.71/0.71, local CV 0.70)


we use random shift and scale to perturb image

here is a tip on ensemble strategy:


we note that lb score is correlated to the number of invalid inchi (verified from rdkit). If you have fewer invalid inchi, you are likely to better LB score.

instead of making many submissions, you can use the number of invalid inchi to estimate your lb score. To know the limit of LB score of your current method, you can extrapolate your graph to the lowest number of possible inchi. If that doesn't work well, you can change your ensemble strategy so that you have a steeper graph. (i.e. better lb score per invalid inchi)

Acknowledgment
I am grateful to Z by HP & NVIDIA for sponsoring me a Z8 Workstation with dual RTX8000 GPU for this competition. In the final days, I need to run TNT/VIT/CAIT ensemble (5 transformers) on the 1.6 million test images. Some of the test images are large and have long sequences. Besides high computative power, we also need high GPU memory.

One RTX8000 GPU has 48GB onboard. It takes about 7 hours to process all images using the workstation. This is very fast and definitely help our team to secure a gold medal of 10th placing in the final ranking!


# PART 2
10th solution & thoughts - Part2. LB 0.61 [ Think out of the box version ]
Part 1 by the great leader @hengck23:
https://www.kaggle.com/c/bms-molecular-translation/discussion/243766 the TNT/VIT/CAIT ensemble model has provided a good baseline for my last-minute shot.

Before technical details
From the competition overview:

Existing tools produce 90% accuracy but only under optimal conditions. Historical sources often have some level of image corruption, which reduces performance to near zero.

I come from the so said existing tools world with zero deep learning experience. I joined the competition to prove this overview is not true (but also true). My colleagues are the developers of MolVec: https://molvec.ncats.io. "Traditional OCR method is able to achieve >90% accuracy on near-perfect images", but the first submission slapped my face hard: LD ~ 80, just slightly better than the H2O naïve baseline. Some heuristic fine tuning (link disconnected components, atomic/bond fixers) has made molvec perform much better on noised images, but we found that pure OCR method stuck on the LD = 5 line. Improvement is possible but it is limited by "what is presented by the image" rather than "what is encoded in the image". So I wrote about ghost atom problem in image resizing. In addition, OCR methods which heavily rely on heuristics (e.g., the appearance of atom/bond/connection) has a hard time dealing with complex cross-bond structure (example). I will discuss about OCR method in the thoughts section.

So, while top teams are approaching LD ~ 1, what's the use of traditional OCR method which is stuck on LD = 5? We abandoned our arrogance and shifted overall strategy: patch deep learning predictions for what it doesn't work well.

by feeding what OCR is good at (near-perfect image)
on the areas where other method doesn't work (big image)
borrow the idea (aka, build molecule atom by atom using chemistry rules)
1. Feed OCR with super-resolution image
"Traditional OCR method is able to achieve >90% accuracy on near-perfect images".

Training	248k randomly selected kaggle image - Indigo rendered image pairs
Model	ResNet34 + U-Net with self-attention fastai tutorial
Image size	832 x 832
Inference	- small image (any dimension < 750): upscale to 832. big image (any dimension 750-1200): upscale to 1248. xbig image (any dimension > 1200): upscale to 1504
InChI generator	super resolution images -> molvec -> .mol -> InChI
Performance	LD ~ 3
Fail	complex molecule
2. Use OCR on extra large image (e.g., width > 1000 pixels)
From the discussion board, there are lots of complaints like "my model doesn't work well on large images". But OCR works decently regardless of image size (10,000 pixels is fine)

No training. Just feed super resolution images to molvec -> .mol -> InChI
I didn't benchmark the performance for big images, but with 90% accuracy, it should outperforms deep learning model that downscaled to 384 or 224 in which most molecular features are lost.

3. Object detection route to build complex molecules from scratch
Idea identical to the DACON winning solution, but to build a valid molecule with acceptable layout from the detected objects on noised image requires lots of "fixers" as well as atom/bond imputation. We introduced the atom group idea to help the valency identification: e.g., CH2 and NH are different than C and N. We separate implicit carbon (shown as vertex) and explicit carbon (shown as Cxx). Meanwhile we enlarge the bbox a bit during the training to include some local context, e.g., a CH2 group usually has 2 bond tips in the bbox. There are totally 54 atom groups and bond types in the 1.6M training set, and we selected 49 for training.

Training	691k training set images. Bounding boxes of atom/atom group/bond were obtained from Indigo rendered svg and custom parser. These 691k includes hard images with a. cross-bond structure or b. atom-atom close contact or c. rare atom/atom groups (e.g, isotope, explicit H, explicit CH2, etc)
Model	EfficientDet-D3
Image size	896 x 896 (bigger the better)
Augmentation	rotation / flip / salt & pepper
InChI generator	bbox -> custom molecular builder script -> .mol -> InChI
Performance	overall mAP=0.87. LD~0.65 for simple molecule. LD~3.5 for complex molecule
Fail	big images dim > 896
Manually check several hundreds hard molecules in the validation set where VIT failed to predict valid InChI, and found that on average MolBuild could reduce LD by ~25 even if MolBuild predicted the wrong molecule. Performance could be better if we have more time on training. It is almost a last-minute model.


4. Merge with TNT/VIT/CAIT ensemble model
LB 0.71 -> 0.61 by updating only 7,679 invalid InChIs (totally 9,800) from VIT in the final submission. Criteria of substitution:

Top tier: the formula component of InChI (the C3H4O2 part) matches the VIT prediction.
Second tier: OCR/MolBuilder-to-VIT edit distance is lower than 10.
Third tier: OCR matches MolBuilder prediction
Finally, push in all OCR result for big and xbig images, regardless of distance
Replacing 0.48% test set gains 0.1 overall LB score, indicating that on average each substitution reduces Levenshtein distance by 21.

Thanks
First I'd like to thank leader @hengck23 and members @mathurinache @ruchi798 who have worked very hard in the past few months. Special thank to @DrHB's team who crashed my day dream that OCR is undefeatable early in this competition. Also thank my colleagues who developed SOTA OCR method and fine tune the parameters. Last but not least, it's been a great & tough competition. Thank you organizers & staff behind the scene.

Thoughts
End of molecular OCR?
With more training data augmented with different renderer parameters, I am convinced that deep learning will win with a big margin, at least for "plain" & small molecule. On the other hand, traditional OCR method which is more generalized can be complementary to SOTA DL model when training data is unreachable (e.g., complex layout). There is always a sweet spot between domain knowledge which offers heuristics and data science which offers precision. Domain knowledge provides indispensable shortcut to the core of problem, and is cost-efficient in pre/post-processing and when big & clean data / annotation are lacking.

Why InChI validation by RDKit works?
First we need to understand what is being learned:

chemistry rules (e.g., a carbon cannot have 10 bonds. InChI string has strict rules)
layout (for 99.9% cases, a benzene ring will be rendered as a hexagon)
noise (vertical/horizontal lines are more likely to disappear in resizing)
synthetic feasibility (wild structure could hardly be synthesized)
From my understanding, most image-driven, encoder-decoder model is good at learning layout (#2) & noise (#3). InChI validation introduces the chemistry rules (#1) elegantly. There is a great discussion here by @nofreewill. On the other hand, OCR method which mainly looks at chemistry will be defeated when noise / layout is out of its domain. The message is that when we understand the problem better, we will solve it more efficiently.

Happy think out of the box.



# HengCK23 starter pack
[completed] transformer starter kit ... in pytorch
this is my terribly slow implementation, i couldn't make a submission yet. Anyway, it shows how to use transformer for a beginner …

i manage to speed up. resnet101d+transformer achieves LB of 3.92, taking 3h to predict 1.6 million test images with 4x Ti1080. The local CV is 3.19 (i train more iterations than the intermediate models in google drive)

speed now is "using fairseq+jit", you can do inference at 2 min for 10_000 images with single GPU.

224x224 images
pre-norm activated trnasformer are used. transformer layer are coded from scrtach.
resnet101d+transformer : performance: 3.3 CV with 40_000 images , in 78 min
resnet26d+transformer : performance: ~3.7 CV estimated
[2] resnet26d+attention+LSTM and [1]resnet26d+LSTM are also included for your study and experiment
train/validation log files, loss curves and intermediate trained models are included
i first trained the image encoder using attention+lstm. the pretained model is then used in the transformer to save time. i not sure if this affect performance
no augmentation in training.
rotation prediction in inference
[1] Show and Tell: A Neural Image Caption Generator
https://arxiv.org/abs/1411.4555

[2] Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
https://arxiv.org/abs/1502.03044

all files are at google drive: https://drive.google.com/drive/folders/1dTfmZxDkDkrnRzz5DOkPOYr9oDIBcrq5?usp=sharing

[2021-apr-06]


please refer to readme.ppt to run the software and experiments
[2021-apr-06a]

dirty code for torch.jit.script model
if i sort the validation samples by length, i can complete 40_000 test samples in 18 min
[2021-apr-07a]

code to migrate from my transformer to fairseq transformer.
using fairseq+jit, you can do interence at 2 min for 10_000 images.
fairseq uses cache for key, values.
[2021-apr-07b]

code to to train TNT(transformer in transformer) image encoder+ token transformer decoder
80 min to do inference for 1.6 millions image with 4xTi1080. CV =1.9, LB =3.1
use fairseq API
you can modify 224 input to 320 to get CV=1.4, LB2.0 !!!
[2021-apr-24]

implement patch-based input for transformer encoder. Please referto th PPT for deails.
you learn how to prepare and store image as patches
how to modify image transformer to accept variable-length input of patches. How to modify position encoding for patch input.
how to set the mask values for transformer encoder and decoder
a small trained model (using 0.8 input scale) and training log is provided. the results of patched based image transformer is the similar to image one, but running much faster (CV-teacher forcing: 1.27, CV-without teacher forcing : about 1.35 for 40000 validation set). time: 5.5 min for 40000 validation set, 55 min for 400000 test images (25% of all). LB = 2.01
there is no submission code or jit inference. This is left as an exercise for you. It is easy to modify the code
[bug] as i was running the submission code, i note that the some test images have larger num of patch than the train, this affects the positional encoding for the input patch.

[2021-apr-25]

model py file for the original VIT vision transformer. you can use this to replace the TNT. There is no pretrain model for deep TNT. But there are larger and deeper pretrain model for VIT
Note: in some way, it is similar to this keras code:
https://www.kaggle.com/aditya08/imagecaptioning-show-attnd-tell-w-transformer