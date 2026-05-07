## Page 1

* [comment] 这个是不是有点绝对

  [source] mathematically valid 10at arbitrary local strain

* [comment] 这个也不要了吧，最终的keyword可以是： deformable image registration; brain MRI; Transformer-based registration; Jacobian determinant; Jacobian-aware regularization; hyperelastic regularization; folding suppression; deformation regularity

  [source] deformable registration

* [comment] 换成deformable image registration

  [source] medical image registration

* [comment] 我认为把他和transformer一起删掉，换成Transformer-based registration

  [source] HypEReg-TransMorph; 

## Page 2

* [comment] 这一整段在和后面discussion有点重复表述，我觉得可以将整段缩减一下

  [source] Clinical motivation. The need for diffeomorphic, fold-free deformations is not a purely 44mathematical concern: in clinical neuroimaging, the Jacobian determinant of the registration 45field is itself the quantity reported in tensor-based morphometry (TBM) and longitudinal 46atrophy/expansion studies for Alzheimer’s disease, mild cognitive impairment, multiple 47sclerosis, normal-pressure hydrocephalus, and post-surgical follow-up of brain tumors and 48epilepsy resections. When local foldings ( det Jϕ ≤ 0) appear in the deformation field, the 49corresponding voxels report negative or undefined volume change, biasing voxel-wise 50statistical maps and inflating false-p

* [comment] 说的可能有点绝对

  [source] s exact at arbitrarily large local 66strains, and supplies a per-voxel quadratic supervision signal that activates on every 67folded voxel;

## Page 3

* [comment] 这其实算不上贡献

  [source] An open reproducibility package. Training/evaluation code, configurations, per-case 86metric exports, figure scripts, and the trained checkpoint manifest are released openly 87so that any third party can retrain HypEReg-TransMorph, regenerate every table and 88figure, and integrate the loss into other dense-flow backbones without architectural 89changes

* [comment] 这里写的是你的设置了，不应该放在这个地方

  [source] In this manuscript, CoTr, nnFormer, and 106PVT are treated as adapted Transformer-family backbones used in the same IXI evaluation 107workflow [ 14– 16 ]. For each backbone, the registration adapter concatenates moving/fixed 108images as a two-channel input and uses the repository flow-prediction decoder head 109to output (Iw, ϕ); all three adapters are evaluated under the same atlas-to-subject proto- 110col. The objective here is not to propose a new Transformer architecture, but to evaluate 111training-time regularization on a TransMorph-style dense-flow network. 112

## Page 5

* [comment] 是不是漏了引用

  [source]  MIDIR;

## Page 7

* [comment] 这里和3.3末端重复，其实可以只在这里提出来就行了，3.3只保留框架概述和figure1引出就行了 而且你里面的hypereg这个损失项感觉没必要，你的设置应该就是1.0 

  [source] The deformation is defined as ϕ(x) = x + u(x), where u : Ω → R3 is the dense 203displacement field. The unified training objective is 204Ltotal = Lsim + λgradLgrad + λHypERegαLlength + βLvolume + γLfold

## Page 9

* [comment] NSD@1mm在主稿Table 3里没有出现

  [source] NSD@1 m

## Page 10

* [comment] 这个可没得地方找，得说明在哪

  [source] ending energy decreases from 46.0976 to 33918.6155

* [comment] 补充材料是0.5898

  [source] (q = 0.592);

## Page 11

* [comment] 这个是否有

  [source] corresponding coarse validation-set behavior is reported in the Supplementary 355Materials. 356

## Page 15

* [comment] 3和4这两个图可以再放两个case

  [source] Figure 4. Deformation-grid visualization derived from predicted displacement fields on the same IXIsubject. Cyan lattice lines represent transformed coordinates after warping by each model. Smooth,non-self-intersecting grids indicate stable local geometry, whereas abrupt kinks or crossings indicatefolding risk. In this representative case, HypEReg-TransMorph qualitatively suggests smootherglobal transitions and fewer folding-prone patterns. (HER-TransMorph in figure panels denotes

