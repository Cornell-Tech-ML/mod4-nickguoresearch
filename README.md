# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py


============================
MNIST MULTICLASS

Epoch 1 loss 2.3109305379758833 valid acc 0/16
Epoch 1 loss 11.109288827443198 valid acc 3/16
Epoch 1 loss 5.144833391450941 valid acc 10/16
Epoch 1 loss 3.573764572523622 valid acc 13/16
Epoch 1 loss 2.265581229184604 valid acc 14/16
Epoch 1 loss 3.679090357672684 valid acc 14/16
Epoch 2 loss 0.4678553819922881 valid acc 14/16
Epoch 2 loss 1.8129763149242495 valid acc 14/16
Epoch 2 loss 1.9174558028306925 valid acc 12/16
Epoch 2 loss 1.5364742656606742 valid acc 14/16
Epoch 2 loss 1.6078319433260813 valid acc 14/16
Epoch 2 loss 2.574412171489812 valid acc 13/16
Epoch 3 loss 0.3832975005250196 valid acc 13/16
Epoch 3 loss 2.134048499545812 valid acc 15/16
Epoch 3 loss 1.6793907576796028 valid acc 12/16
Epoch 3 loss 1.3972829511642222 valid acc 15/16
Epoch 3 loss 1.1830268006994007 valid acc 15/16
Epoch 3 loss 3.17935536325742 valid acc 13/16
Epoch 4 loss 0.0589256903383881 valid acc 13/16
Epoch 4 loss 1.6740397469408559 valid acc 15/16
Epoch 4 loss 1.996697496995116 valid acc 13/16
Epoch 4 loss 1.1781506692565527 valid acc 14/16
Epoch 4 loss 1.2713587169825735 valid acc 15/16
Epoch 4 loss 1.6768166211805606 valid acc 14/16
Epoch 5 loss 0.06262807388829249 valid acc 14/16
Epoch 5 loss 1.0964209852710107 valid acc 15/16
Epoch 5 loss 1.3937572421398647 valid acc 14/16
Epoch 5 loss 0.7929084103851336 valid acc 15/16
Epoch 5 loss 0.6368566166730736 valid acc 16/16
Epoch 5 loss 1.1489959668113785 valid acc 16/16
Epoch 6 loss 0.013894761901823269 valid acc 16/16
Epoch 6 loss 0.9199489006816488 valid acc 15/16
Epoch 6 loss 1.6687101001815599 valid acc 14/16
Epoch 6 loss 0.40292262662056144 valid acc 16/16
Epoch 6 loss 0.8091652620007166 valid acc 16/16
Epoch 6 loss 1.018677154323462 valid acc 15/16
Epoch 7 loss 0.035791466359026254 valid acc 15/16
Epoch 7 loss 0.5641338450664483 valid acc 16/16
Epoch 7 loss 1.38137985181622 valid acc 15/16
Epoch 7 loss 0.5670979780867491 valid acc 14/16
Epoch 7 loss 0.7317010889932225 valid acc 16/16
Epoch 7 loss 0.5435857682766874 valid acc 16/16
Epoch 8 loss 0.01578945382614046 valid acc 16/16
Epoch 8 loss 0.6296457855426549 valid acc 16/16
Epoch 8 loss 1.1917808281027846 valid acc 15/16
Epoch 8 loss 0.2520212649942224 valid acc 16/16
Epoch 8 loss 0.850756863456411 valid acc 16/16
Epoch 8 loss 0.6643179094893995 valid acc 15/16
Epoch 9 loss 0.013784752154030314 valid acc 16/16
Epoch 9 loss 0.465043483428785 valid acc 15/16
Epoch 9 loss 1.3249261713973783 valid acc 15/16
Epoch 9 loss 0.3881588413230394 valid acc 16/16
Epoch 9 loss 0.2670070969866555 valid acc 16/16
Epoch 9 loss 0.4881009909667391 valid acc 16/16
Epoch 10 loss 0.003101873613524135 valid acc 16/16
Epoch 10 loss 0.4275374266104306 valid acc 15/16
Epoch 10 loss 1.2982680785240786 valid acc 15/16
Epoch 10 loss 0.28873282323964045 valid acc 15/16
Epoch 10 loss 0.7489183561295994 valid acc 16/16
Epoch 10 loss 0.5554763298343739 valid acc 16/16
Epoch 11 loss 0.02506237645764724 valid acc 16/16
Epoch 11 loss 0.30972734221729703 valid acc 16/16
Epoch 11 loss 1.1803198781526443 valid acc 15/16
Epoch 11 loss 0.3532065423695479 valid acc 16/16
Epoch 11 loss 0.5855163964505973 valid acc 16/16
Epoch 11 loss 0.5106548701432068 valid acc 16/16
Epoch 12 loss 0.04894139708346188 valid acc 16/16
Epoch 12 loss 0.46688136975166394 valid acc 16/16
Epoch 12 loss 1.1834532993594182 valid acc 15/16
Epoch 12 loss 0.26802085184092445 valid acc 16/16
Epoch 12 loss 0.5509216250663385 valid acc 16/16
Epoch 12 loss 0.6520468655888775 valid acc 16/16
Epoch 13 loss 0.0017379009727829895 valid acc 16/16
Epoch 13 loss 0.6386318938293575 valid acc 16/16
Epoch 13 loss 0.9432524278856063 valid acc 14/16
Epoch 13 loss 0.14672317893477224 valid acc 16/16
Epoch 13 loss 0.8062500713897827 valid acc 16/16
Epoch 13 loss 0.5291415699124795 valid acc 16/16
Epoch 14 loss 0.06985193634499257 valid acc 16/16
Epoch 14 loss 0.5493109108658694 valid acc 15/16
Epoch 14 loss 1.3255263295445654 valid acc 15/16
Epoch 14 loss 0.16931543981291014 valid acc 15/16
Epoch 14 loss 0.34684761027626243 valid acc 15/16
Epoch 14 loss 0.8333862005414836 valid acc 15/16
Epoch 15 loss 0.019894691659135478 valid acc 15/16
Epoch 15 loss 0.4681361353336192 valid acc 15/16
Epoch 15 loss 1.0469205682393072 valid acc 15/16
Epoch 15 loss 0.3604502429858682 valid acc 15/16
Epoch 15 loss 0.40556880791903827 valid acc 16/16
Epoch 15 loss 0.3088907679044256 valid acc 15/16
Epoch 16 loss 0.000880121598409554 valid acc 15/16
Epoch 16 loss 0.5458160838932998 valid acc 16/16
Epoch 16 loss 1.2311456709563435 valid acc 16/16
Epoch 16 loss 0.2320357029089043 valid acc 16/16
Epoch 16 loss 0.29858074106279575 valid acc 16/16
Epoch 16 loss 0.42384917500892094 valid acc 16/16
Epoch 16 loss 0.41578568239266644 valid acc 15/16
Epoch 16 loss 0.17205812035650994 valid acc 15/16
Epoch 16 loss 0.5159589445880025 valid acc 16/16
Epoch 16 loss 0.22853866760240826 valid acc 15/16
Epoch 17 loss 0.01201939971408456 valid acc 15/16
Epoch 17 loss 0.24237475409033726 valid acc 15/16
Epoch 17 loss 0.8072896615115901 valid acc 15/16
Epoch 17 loss 0.3235886026631968 valid acc 15/16
Epoch 17 loss 0.37376334679763357 valid acc 15/16
Epoch 17 loss 0.4003963592903971 valid acc 16/16
Epoch 17 loss 0.4572106845688326 valid acc 15/16
Epoch 17 loss 1.1658259272797067 valid acc 14/16
Epoch 17 loss 0.31667138473387074 valid acc 14/16
Epoch 17 loss 0.14186112236618345 valid acc 15/16
Epoch 17 loss 0.9253163188315421 valid acc 15/16
Epoch 17 loss 1.5272894983119985 valid acc 15/16
Epoch 17 loss 0.1742149798204471 valid acc 14/16
Epoch 17 loss 0.6548449023018735 valid acc 14/16
Epoch 17 loss 0.40709556689159154 valid acc 14/16
Epoch 17 loss 0.47305740100577365 valid acc 15/16
Epoch 17 loss 0.8094874588401985 valid acc 14/16
Epoch 17 loss 0.8690177430673905 valid acc 15/16
Epoch 17 loss 0.8071883341417351 valid acc 15/16
Epoch 17 loss 0.33892083496023295 valid acc 16/16
Epoch 17 loss 1.0294965921235728 valid acc 15/16
Epoch 17 loss 0.5503603199393624 valid acc 15/16
Epoch 17 loss 0.12062555451402879 valid acc 15/16
Epoch 17 loss 0.310691580717086 valid acc 15/16
Epoch 17 loss 0.4143792976336903 valid acc 15/16
Epoch 17 loss 0.5189886362639435 valid acc 15/16
Epoch 17 loss 0.19047819365076038 valid acc 14/16
Epoch 17 loss 0.4048028671447552 valid acc 14/16
Epoch 17 loss 0.3864452707994944 valid acc 15/16
Epoch 17 loss 0.07341762252746646 valid acc 15/16
Epoch 17 loss 0.17822824889633143 valid acc 15/16
Epoch 17 loss 0.2411792486764606 valid acc 15/16
Epoch 17 loss 0.0768239618309493 valid acc 15/16
Epoch 17 loss 0.4223541500014213 valid acc 16/16
Epoch 17 loss 1.8394338316606413 valid acc 15/16
Epoch 17 loss 0.576516229499733 valid acc 15/16
Epoch 17 loss 0.44878282053662333 valid acc 16/16
Epoch 17 loss 0.7406263147079369 valid acc 15/16
Epoch 17 loss 0.7530445503778507 valid acc 15/16
Epoch 17 loss 0.5311400352741903 valid acc 15/16
Epoch 17 loss 0.2971627303835892 valid acc 15/16
Epoch 17 loss 0.5725533230880386 valid acc 15/16
Epoch 17 loss 0.6543814370294614 valid acc 15/16
Epoch 17 loss 0.062337059584492885 valid acc 15/16
Epoch 17 loss 1.5275404202327694 valid acc 15/16
Epoch 17 loss 0.15518544766097467 valid acc 15/16
Epoch 17 loss 0.6057200109533547 valid acc 15/16
Epoch 17 loss 0.6034560769746755 valid acc 15/16
Epoch 17 loss 0.36397128422297925 valid acc 15/16
Epoch 17 loss 0.4597808458878202 valid acc 15/16
Epoch 17 loss 0.3480628843458722 valid acc 15/16
Epoch 17 loss 0.8412367212432437 valid acc 16/16
Epoch 17 loss 0.6112331784771511 valid acc 15/16
Epoch 17 loss 0.08052532335088952 valid acc 15/16
Epoch 17 loss 0.8327796537077785 valid acc 15/16
Epoch 17 loss 0.9018254113478461 valid acc 15/16
Epoch 17 loss 0.5691537460729695 valid acc 15/16
Epoch 17 loss 0.7790429159708424 valid acc 15/16
Epoch 17 loss 0.7422844103700407 valid acc 15/16
Epoch 17 loss 0.4145610648509399 valid acc 16/16
Epoch 17 loss 0.44333240638058424 valid acc 16/16
Epoch 17 loss 0.5139312209469363 valid acc 16/16
Epoch 17 loss 0.644305618250814 valid acc 16/16
Epoch 18 loss 0.005590917372662047 valid acc 16/16
Epoch 18 loss 0.522117473866371 valid acc 15/16
Epoch 18 loss 0.5752210120173823 valid acc 15/16
Epoch 18 loss 0.4248477722954527 valid acc 15/16
Epoch 18 loss 0.6781325044980969 valid acc 15/16
Epoch 18 loss 0.727931496907835 valid acc 15/16
Epoch 18 loss 0.47004521852237213 valid acc 14/16
Epoch 18 loss 1.033661165624031 valid acc 15/16
Epoch 18 loss 1.0333796198080223 valid acc 15/16
Epoch 18 loss 0.16410304352967942 valid acc 15/16
Epoch 18 loss 0.5687627459450267 valid acc 15/16
Epoch 18 loss 1.4819734547756807 valid acc 15/16
Epoch 18 loss 0.6442185016572872 valid acc 14/16
Epoch 18 loss 0.8124156224225356 valid acc 15/16
Epoch 18 loss 0.3693433202371622 valid acc 15/16
Epoch 18 loss 0.5119063294320341 valid acc 15/16
Epoch 18 loss 0.8191084514183612 valid acc 14/16
Epoch 18 loss 0.8852429045657894 valid acc 15/16
Epoch 18 loss 0.5415249439838207 valid acc 14/16
Epoch 18 loss 0.5044317959182053 valid acc 14/16
Epoch 18 loss 1.2168353897701938 valid acc 14/16
Epoch 18 loss 0.7661911444718372 valid acc 14/16
Epoch 18 loss 0.10386889041992094 valid acc 15/16
Epoch 18 loss 0.3392634317894713 valid acc 15/16
Epoch 18 loss 0.5744354094969889 valid acc 15/16
Epoch 18 loss 0.8472969063404641 valid acc 15/16
Epoch 18 loss 0.4036606618146781 valid acc 15/16
Epoch 18 loss 0.2800081345091761 valid acc 15/16
Epoch 18 loss 0.39231925523381933 valid acc 15/16
Epoch 18 loss 0.36827223053609837 valid acc 16/16
Epoch 18 loss 0.09986765574592893 valid acc 16/16
Epoch 18 loss 0.3216998785016702 valid acc 15/16
Epoch 18 loss 0.2871185844621115 valid acc 16/16
Epoch 18 loss 0.331232489188391 valid acc 16/16
Epoch 18 loss 1.8079407001934702 valid acc 16/16
Epoch 18 loss 0.47844935258499255 valid acc 16/16
Epoch 18 loss 0.4314190813870602 valid acc 16/16
Epoch 18 loss 0.6884080466093867 valid acc 16/16
Epoch 18 loss 0.5527562221205363 valid acc 16/16
Epoch 18 loss 0.28263161544751303 valid acc 16/16
Epoch 18 loss 0.20085121810375367 valid acc 16/16
Epoch 18 loss 1.1267918562657728 valid acc 16/16
Epoch 18 loss 0.6493724232276366 valid acc 15/16
Epoch 18 loss 0.5504973858896238 valid acc 15/16
Epoch 18 loss 1.1327359944138946 valid acc 15/16
Epoch 18 loss 0.03352849524819734 valid acc 15/16
Epoch 18 loss 0.4413461606011029 valid acc 16/16
Epoch 18 loss 0.6644357453942149 valid acc 15/16
Epoch 18 loss 0.5105070198586478 valid acc 15/16
Epoch 18 loss 0.18332198828349186 valid acc 16/16
Epoch 18 loss 0.3374046868387852 valid acc 15/16
Epoch 18 loss 0.3017624122932167 valid acc 15/16
Epoch 18 loss 0.8177361926384715 valid acc 15/16
Epoch 18 loss 0.10612484964818765 valid acc 15/16
Epoch 18 loss 0.6578736154971532 valid acc 15/16
Epoch 18 loss 0.33044974432055185 valid acc 15/16
Epoch 18 loss 1.229800914905703 valid acc 15/16
Epoch 18 loss 0.5683103020054294 valid acc 15/16
Epoch 18 loss 0.32596107308984135 valid acc 15/16
Epoch 18 loss 0.6085297420628941 valid acc 16/16
Epoch 18 loss 0.24883570644811442 valid acc 15/16
Epoch 18 loss 0.38102327155296334 valid acc 15/16
Epoch 18 loss 0.06322067509197599 valid acc 15/16
Epoch 19 loss 0.0017487479204294802 valid acc 15/16
Epoch 19 loss 0.4834103149073544 valid acc 16/16
Epoch 19 loss 1.079584538037631 valid acc 16/16
Epoch 19 loss 1.1438078562448921 valid acc 15/16
Epoch 19 loss 0.5192863378357974 valid acc 15/16
Epoch 19 loss 0.4559225218910851 valid acc 16/16
Epoch 19 loss 0.5431845550062002 valid acc 15/16
Epoch 19 loss 0.8229586228113168 valid acc 16/16
Epoch 19 loss 0.7317533984160365 valid acc 16/16
Epoch 19 loss 0.3278170732299326 valid acc 16/16
Epoch 19 loss 0.6413904264301693 valid acc 16/16
Epoch 19 loss 1.6211609021359705 valid acc 16/16
Epoch 19 loss 0.4834101323740591 valid acc 15/16
Epoch 19 loss 1.1414550811464552 valid acc 15/16
Epoch 19 loss 0.622645575121113 valid acc 15/16
Epoch 19 loss 0.36467650279072145 valid acc 15/16
Epoch 19 loss 0.516312293018672 valid acc 16/16
Epoch 19 loss 0.6924618771118165 valid acc 16/16
Epoch 19 loss 0.6264002208973559 valid acc 15/16
Epoch 19 loss 0.46972114105108004 valid acc 16/16
Epoch 19 loss 1.0261447787828892 valid acc 15/16
Epoch 19 loss 0.1529613727437766 valid acc 15/16
Epoch 19 loss 0.12826403308909357 valid acc 15/16
Epoch 19 loss 0.3995039260038758 valid acc 15/16
Epoch 19 loss 0.34450890677093743 valid acc 15/16
Epoch 19 loss 0.611347417155505 valid acc 16/16
Epoch 19 loss 0.27919249946353336 valid acc 15/16
Epoch 19 loss 0.052296632559372835 valid acc 15/16
Epoch 19 loss 0.4164199855283991 valid acc 15/16
Epoch 19 loss 0.25612329944682877 valid acc 16/16
Epoch 19 loss 0.1197904803530823 valid acc 16/16
Epoch 19 loss 0.23338619379231082 valid acc 16/16
Epoch 19 loss 0.4358011779563783 valid acc 16/16
Epoch 19 loss 0.43313826678583145 valid acc 16/16
Epoch 19 loss 1.514344791590284 valid acc 16/16
Epoch 19 loss 0.43664858616633856 valid acc 16/16
Epoch 19 loss 0.2647187871621764 valid acc 16/16
Epoch 19 loss 0.3496228132401691 valid acc 16/16
Epoch 19 loss 0.7131970927135878 valid acc 15/16
Epoch 19 loss 0.617400621020829 valid acc 15/16
Epoch 19 loss 0.20687546059071654 valid acc 15/16
Epoch 19 loss 0.7137844200009033 valid acc 16/16
Epoch 19 loss 0.32646815362859094 valid acc 15/16
Epoch 19 loss 0.2516166900362691 valid acc 15/16
Epoch 19 loss 0.501284649729852 valid acc 16/16
Epoch 19 loss 0.2588049843193573 valid acc 16/16
Epoch 19 loss 0.9278435554478599 valid acc 16/16
Epoch 19 loss 0.3409264123613242 valid acc 15/16
Epoch 19 loss 0.29415672519041325 valid acc 15/16
Epoch 19 loss 0.2227706192107972 valid acc 15/16
Epoch 19 loss 0.3054208930628349 valid acc 15/16
Epoch 19 loss 0.5728920908315243 valid acc 15/16
Epoch 19 loss 0.14071762390464285 valid acc 15/16
Epoch 19 loss 0.0371913689257331 valid acc 15/16
Epoch 19 loss 0.2166886919731807 valid acc 15/16
Epoch 19 loss 0.0774324586279413 valid acc 15/16
Epoch 19 loss 0.980664910646377 valid acc 15/16
Epoch 19 loss 0.10777059413216988 valid acc 15/16
Epoch 19 loss 0.34738535079751875 valid acc 15/16
Epoch 19 loss 0.3101902825159743 valid acc 15/16
Epoch 19 loss 0.6489303558903083 valid acc 15/16
Epoch 19 loss 0.42926593901351323 valid acc 15/16
Epoch 19 loss 0.6958971262073328 valid acc 16/16
Epoch 20 loss 0.010132429104164361 valid acc 16/16
Epoch 20 loss 0.5145394705965555 valid acc 16/16
Epoch 20 loss 1.053261018971809 valid acc 16/16
Epoch 20 loss 0.6198416246851263 valid acc 15/16
Epoch 20 loss 0.3791257831644057 valid acc 15/16
Epoch 20 loss 0.32432782001893545 valid acc 15/16
Epoch 20 loss 0.5984433542363977 valid acc 14/16
Epoch 20 loss 0.8398831022489893 valid acc 14/16
Epoch 20 loss 0.6190145605282257 valid acc 14/16
Epoch 20 loss 0.5818039444360361 valid acc 14/16
Epoch 20 loss 0.457828106612445 valid acc 14/16
Epoch 20 loss 0.8933661948562774 valid acc 14/16
Epoch 20 loss 0.3170049916691145 valid acc 14/16
Epoch 20 loss 0.5512032585585096 valid acc 14/16
Epoch 20 loss 0.3777955582645807 valid acc 15/16
Epoch 20 loss 0.22840612628975754 valid acc 15/16
Epoch 20 loss 0.6572064335632583 valid acc 14/16
Epoch 20 loss 0.9066835360157541 valid acc 14/16
Epoch 20 loss 0.8659195610349533 valid acc 15/16
Epoch 20 loss 0.2164346815985426 valid acc 15/16
Epoch 20 loss 0.7782653638754145 valid acc 15/16
Epoch 20 loss 0.40565495953794173 valid acc 16/16
Epoch 20 loss 0.12502983953272825 valid acc 15/16
Epoch 20 loss 0.16683896293147943 valid acc 15/16
Epoch 20 loss 0.8358804877768399 valid acc 15/16
Epoch 20 loss 0.49479220193416823 valid acc 16/16
Epoch 20 loss 0.38123815124800486 valid acc 16/16
Epoch 20 loss 0.3674336477859013 valid acc 15/16
Epoch 20 loss 0.321225927535069 valid acc 15/16
Epoch 20 loss 0.1272506716309262 valid acc 15/16
Epoch 20 loss 0.2924395898203993 valid acc 15/16
Epoch 20 loss 0.37661332194225805 valid acc 15/16
Epoch 20 loss 0.14999080202847415 valid acc 15/16
Epoch 20 loss 0.4322552622761767 valid acc 15/16
Epoch 20 loss 1.31155837784808 valid acc 15/16
Epoch 20 loss 0.22372761280941178 valid acc 15/16
Epoch 20 loss 0.4155668891767383 valid acc 16/16
Epoch 20 loss 0.4856981128692161 valid acc 16/16
Epoch 20 loss 0.4234565698078969 valid acc 15/16
Epoch 20 loss 0.16749190719246426 valid acc 16/16
Epoch 20 loss 0.07850484890150962 valid acc 16/16
Epoch 20 loss 0.4968897571952726 valid acc 15/16
Epoch 20 loss 0.37866686274768235 valid acc 15/16
Epoch 20 loss 0.04539270505099277 valid acc 15/16
Epoch 20 loss 1.254227181372884 valid acc 15/16
Epoch 20 loss 0.094394190648882 valid acc 15/16
Epoch 20 loss 0.4229471814421042 valid acc 15/16
Epoch 20 loss 1.0023772834555837 valid acc 15/16
Epoch 20 loss 0.2155650172234902 valid acc 15/16
Epoch 20 loss 0.09435475028749542 valid acc 15/16
Epoch 20 loss 0.09833208702355734 valid acc 15/16
Epoch 20 loss 0.2415706907559099 valid acc 15/16
Epoch 20 loss 0.21196955888040994 valid acc 15/16
Epoch 20 loss 0.10404110225237845 valid acc 15/16
Epoch 20 loss 0.471346168371143 valid acc 15/16
Epoch 20 loss 0.4164545027522528 valid acc 15/16
Epoch 20 loss 0.8201339128924369 valid acc 14/16
Epoch 20 loss 0.10189766717275725 valid acc 14/16
Epoch 20 loss 0.15450820416822314 valid acc 14/16
Epoch 20 loss 0.4033697040624859 valid acc 14/16
Epoch 20 loss 0.15989356858557296 valid acc 14/16
Epoch 20 loss 0.4816645031043608 valid acc 15/16
Epoch 20 loss 0.2091506195168078 valid acc 15/16
Epoch 21 loss 0.0038653059768768137 valid acc 15/16
Epoch 21 loss 0.6354556123422409 valid acc 14/16
Epoch 21 loss 1.1370608813286776 valid acc 15/16
Epoch 21 loss 0.2744184025237682 valid acc 15/16
Epoch 21 loss 0.22743146302993597 valid acc 14/16
Epoch 21 loss 0.49216773509432293 valid acc 15/16
Epoch 21 loss 0.45542166846297694 valid acc 14/16
Epoch 21 loss 0.9117062197643643 valid acc 15/16
Epoch 21 loss 0.3932538127104198 valid acc 15/16
Epoch 21 loss 0.5231598698251304 valid acc 15/16
Epoch 21 loss 0.5323389694998005 valid acc 15/16
Epoch 21 loss 1.7993382051854687 valid acc 14/16
Epoch 21 loss 0.19057438967095705 valid acc 14/16
Epoch 21 loss 1.1850133200775215 valid acc 14/16
Epoch 21 loss 0.36151298336885285 valid acc 14/16
Epoch 21 loss 0.45638711538922294 valid acc 15/16
Epoch 21 loss 1.0041427868552248 valid acc 14/16
Epoch 21 loss 0.893486753519076 valid acc 14/16
Epoch 21 loss 0.9643143274225776 valid acc 14/16
Epoch 21 loss 0.3623529072547601 valid acc 15/16
Epoch 21 loss 0.878321772511079 valid acc 13/16
Epoch 21 loss 0.2814654565146115 valid acc 14/16
Epoch 21 loss 0.07938929136255572 valid acc 15/16
Epoch 21 loss 0.28195688655498513 valid acc 15/16
Epoch 21 loss 0.44579911612995965 valid acc 15/16
Epoch 21 loss 0.30124571018359736 valid acc 15/16
Epoch 21 loss 0.31495040074921954 valid acc 14/16
Epoch 21 loss 0.5578600737480605 valid acc 15/16
Epoch 21 loss 0.3109526550773506 valid acc 15/16
Epoch 21 loss 0.20734903058280005 valid acc 14/16
Epoch 21 loss 0.23542398301380352 valid acc 14/16
Epoch 21 loss 0.3351941585318906 valid acc 15/16
Epoch 21 loss 0.5057655788403423 valid acc 14/16
Epoch 21 loss 0.4943455051896076 valid acc 14/16
Epoch 21 loss 1.876809455141733 valid acc 14/16
Epoch 21 loss 0.7253135640598544 valid acc 16/16
Epoch 21 loss 0.4984698396103656 valid acc 16/16
Epoch 21 loss 0.9006182545285659 valid acc 15/16
Epoch 21 loss 0.25938609482554614 valid acc 16/16
Epoch 21 loss 0.42046499698326434 valid acc 16/16
Epoch 21 loss 0.28696286941724225 valid acc 15/16
Epoch 21 loss 0.3298927514352799 valid acc 16/16
Epoch 21 loss 0.20335732128997533 valid acc 15/16
Epoch 21 loss 0.22229858008952702 valid acc 15/16
Epoch 21 loss 0.6119949295692716 valid acc 16/16
Epoch 21 loss 0.04911943262959642 valid acc 16/16
Epoch 21 loss 0.7203731844712946 valid acc 16/16
Epoch 21 loss 1.498115475809235 valid acc 15/16
Epoch 21 loss 0.21096640506618455 valid acc 15/16
Epoch 21 loss 0.053580831422304326 valid acc 15/16
Epoch 21 loss 0.19248777990166413 valid acc 15/16
Epoch 21 loss 0.408954213043755 valid acc 15/16
Epoch 21 loss 0.3697882464605369 valid acc 16/16
Epoch 21 loss 0.020818880281244526 valid acc 16/16
Epoch 21 loss 0.6782968346410649 valid acc 15/16
Epoch 21 loss 0.1027336689532084 valid acc 15/16
Epoch 21 loss 0.2966808932627039 valid acc 15/16
Epoch 21 loss 0.30325582235791326 valid acc 15/16
Epoch 21 loss 0.4030134925225105 valid acc 15/16
Epoch 21 loss 0.2594412052756956 valid acc 15/16
Epoch 21 loss 0.3197421015234329 valid acc 15/16
Epoch 21 loss 0.4286407438781474 valid acc 15/16
Epoch 21 loss 0.4755234568523673 valid acc 16/16
Epoch 22 loss 0.000882459373348401 valid acc 16/16
Epoch 22 loss 0.24918905094884686 valid acc 15/16
Epoch 22 loss 0.313977520681047 valid acc 15/16
Epoch 22 loss 0.3633957284029075 valid acc 15/16
Epoch 22 loss 0.4172749233448234 valid acc 15/16
Epoch 22 loss 0.5275326101875151 valid acc 15/16
Epoch 22 loss 0.32501200278895076 valid acc 15/16
Epoch 22 loss 0.48076037096611585 valid acc 15/16
Epoch 22 loss 1.0420923432051827 valid acc 15/16
Epoch 22 loss 0.14569314250527593 valid acc 15/16
Epoch 22 loss 0.16184956024857702 valid acc 15/16
Epoch 22 loss 0.5753020453280496 valid acc 15/16
Epoch 22 loss 0.47839464852140073 valid acc 15/16
Epoch 22 loss 1.214288837542107 valid acc 13/16
Epoch 22 loss 0.4822488745520864 valid acc 14/16
Epoch 22 loss 0.28501902030431103 valid acc 14/16
Epoch 22 loss 0.5760506342102301 valid acc 14/16
Epoch 22 loss 0.7335649043853786 valid acc 14/16
Epoch 22 loss 0.5209229040029962 valid acc 14/16
Epoch 22 loss 0.23429608410885414 valid acc 14/16
Epoch 22 loss 1.1725500384201613 valid acc 15/16
Epoch 22 loss 0.2315531180195086 valid acc 14/16
Epoch 22 loss 0.2011163477114541 valid acc 14/16
Epoch 22 loss 0.20635008830833035 valid acc 14/16
Epoch 22 loss 0.7145441517433975 valid acc 15/16
Epoch 22 loss 1.0106604982548213 valid acc 15/16
Epoch 22 loss 0.23996765844639248 valid acc 14/16
Epoch 22 loss 0.08043790141287044 valid acc 15/16
Epoch 22 loss 0.19242034398727775 valid acc 15/16
Epoch 22 loss 0.1929009743078486 valid acc 14/16
Epoch 22 loss 0.07665405438598516 valid acc 15/16
Epoch 22 loss 0.12464011317889596 valid acc 15/16
Epoch 22 loss 0.19904522838696534 valid acc 16/16
Epoch 22 loss 0.35074890114199647 valid acc 15/16
Epoch 22 loss 1.4732515702161715 valid acc 16/16
Epoch 22 loss 0.24322642469625652 valid acc 16/16
Epoch 22 loss 0.17317361751787838 valid acc 16/16
Epoch 22 loss 0.12252036408008807 valid acc 16/16
Epoch 22 loss 0.40080104443118875 valid acc 16/16
Epoch 22 loss 0.7403458869882643 valid acc 16/16
Epoch 22 loss 0.24907074485044675 valid acc 15/16
Epoch 22 loss 0.3171002927362904 valid acc 16/16
Epoch 22 loss 0.3423856721715597 valid acc 15/16
Epoch 22 loss 0.21498471318891427 valid acc 15/16
Epoch 22 loss 1.2086705082399805 valid acc 15/16
Epoch 22 loss 0.44774513215272954 valid acc 15/16
Epoch 22 loss 0.6679939553996124 valid acc 15/16
Epoch 22 loss 0.6296734032107982 valid acc 15/16
Epoch 22 loss 0.12241638652817421 valid acc 15/16
Epoch 22 loss 0.1763527823916884 valid acc 16/16
Epoch 22 loss 0.04882477321070407 valid acc 16/16
Epoch 22 loss 0.2960153593888158 valid acc 16/16
Epoch 22 loss 0.4677437964304784 valid acc 16/16
Epoch 22 loss 0.038618431867409375 valid acc 16/16
Epoch 22 loss 0.29420528975481647 valid acc 15/16
Epoch 22 loss 0.26203155293581626 valid acc 15/16
Epoch 22 loss 0.6592644585571071 valid acc 15/16
Epoch 22 loss 0.10217585369241207 valid acc 15/16
Epoch 22 loss 0.3926888216860673 valid acc 15/16
Epoch 22 loss 0.2980460248927298 valid acc 15/16
Epoch 22 loss 0.07663608498053351 valid acc 15/16
Epoch 22 loss 0.08088130981686462 valid acc 15/16
Epoch 22 loss 0.5951170232376903 valid acc 16/16
Epoch 23 loss 0.0005742830314975635 valid acc 15/16
Epoch 23 loss 0.8938526449826587 valid acc 14/16
Epoch 23 loss 0.4453539451696779 valid acc 15/16
Epoch 23 loss 0.2553116982943213 valid acc 15/16
Epoch 23 loss 0.31907291238251023 valid acc 15/16
Epoch 23 loss 0.31213453268793595 valid acc 16/16
Epoch 23 loss 0.26223918601869495 valid acc 16/16
Epoch 23 loss 0.3860517487474666 valid acc 15/16
Epoch 23 loss 0.8650154365970291 valid acc 15/16
Epoch 23 loss 0.27265892312992546 valid acc 15/16
Epoch 23 loss 0.5964892257514394 valid acc 16/16
Epoch 23 loss 0.9696479192808172 valid acc 15/16
Epoch 23 loss 0.3295595373173844 valid acc 15/16
Epoch 23 loss 0.6276630476053686 valid acc 15/16
Epoch 23 loss 0.41661296006726195 valid acc 15/16
Epoch 23 loss 0.23664277088756158 valid acc 15/16
Epoch 23 loss 1.0144502975462144 valid acc 15/16
Epoch 23 loss 0.5051789807541585 valid acc 15/16
Epoch 23 loss 0.42567913471004104 valid acc 15/16
Epoch 23 loss 0.175726071360948 valid acc 15/16
Epoch 23 loss 0.4173819857461808 valid acc 15/16
Epoch 23 loss 0.1432371902732329 valid acc 16/16
Epoch 23 loss 0.03642969912963489 valid acc 15/16
Epoch 23 loss 0.259030130583905 valid acc 16/16
Epoch 23 loss 0.4248174555848565 valid acc 15/16
Epoch 23 loss 0.40596000728104975 valid acc 16/16
Epoch 23 loss 0.44802600551956406 valid acc 16/16
Epoch 23 loss 0.2307576032656475 valid acc 15/16
Epoch 23 loss 0.18379450662211833 valid acc 15/16
Epoch 23 loss 0.04753583866646004 valid acc 15/16
Epoch 23 loss 0.09978329114411812 valid acc 15/16
Epoch 23 loss 0.33974850710516047 valid acc 15/16
Epoch 23 loss 0.47721513281893885 valid acc 14/16
Epoch 23 loss 0.7065979805160232 valid acc 15/16
Epoch 23 loss 0.857886492644111 valid acc 15/16
Epoch 23 loss 0.8119503098665217 valid acc 15/16
Epoch 23 loss 0.4044129583610508 valid acc 16/16
Epoch 23 loss 0.41966909185130774 valid acc 15/16
Epoch 23 loss 0.6225197326676044 valid acc 15/16
Epoch 23 loss 0.6802203756243017 valid acc 16/16
Epoch 23 loss 0.21666488885149404 valid acc 15/16
Epoch 23 loss 1.1438800238540003 valid acc 16/16
Epoch 23 loss 0.5153907979208553 valid acc 15/16
Epoch 23 loss 0.3150569591499629 valid acc 15/16
Epoch 23 loss 0.8301041480496072 valid acc 15/16
Epoch 23 loss 0.057304481044375466 valid acc 15/16
Epoch 23 loss 1.083063673132155 valid acc 15/16
Epoch 23 loss 0.6952682272893506 valid acc 16/16
Epoch 23 loss 0.5767060607599794 valid acc 16/16
Epoch 23 loss 0.17728544529504933 valid acc 16/16
Epoch 23 loss 0.10145204010585546 valid acc 16/16
Epoch 23 loss 0.5901373046766951 valid acc 16/16
Epoch 23 loss 0.575113583593883 valid acc 16/16
Epoch 23 loss 0.010781568680924789 valid acc 16/16
Epoch 23 loss 0.3515750953009101 valid acc 15/16
Epoch 23 loss 0.09171483051189237 valid acc 15/16
Epoch 23 loss 0.42238300822888614 valid acc 16/16
Epoch 23 loss 0.5472521838367721 valid acc 16/16
Epoch 23 loss 0.23311704518589355 valid acc 16/16
Epoch 23 loss 0.042456974869710014 valid acc 16/16
Epoch 23 loss 0.11436159712851504 valid acc 16/16
Epoch 23 loss 0.11840218014881826 valid acc 16/16
Epoch 23 loss 0.15035918659809172 valid acc 16/16
Epoch 24 loss 0.0007392156741602328 valid acc 16/16
Epoch 24 loss 1.1547052466553405 valid acc 16/16
Epoch 24 loss 0.6935602342253429 valid acc 16/16
Epoch 24 loss 0.3144654004831595 valid acc 15/16
Epoch 24 loss 0.30478152923068746 valid acc 16/16
Epoch 24 loss 0.4181819962326947 valid acc 16/16
Epoch 24 loss 0.9855498128163204 valid acc 13/16
Epoch 24 loss 1.3494635183822763 valid acc 16/16
Epoch 24 loss 1.0623943120675572 valid acc 15/16
Epoch 24 loss 0.2835482696199775 valid acc 15/16
Epoch 24 loss 0.3800205855611611 valid acc 15/16
Epoch 24 loss 0.44392392175884654 valid acc 16/16
Epoch 24 loss 0.4118909955565348 valid acc 16/16
Epoch 24 loss 0.4497843543506481 valid acc 16/16
Epoch 24 loss 0.3069517162828858 valid acc 16/16
Epoch 24 loss 0.3538232018208425 valid acc 15/16
Epoch 24 loss 0.6156374814359806 valid acc 16/16
Epoch 24 loss 0.4858887072947227 valid acc 15/16
Epoch 24 loss 0.39636812659744236 valid acc 16/16
Epoch 24 loss 0.3266358156272864 valid acc 16/16
Epoch 24 loss 0.9140378692199903 valid acc 16/16
Epoch 24 loss 0.5759909476186533 valid acc 16/16
Epoch 24 loss 0.14955198335395758 valid acc 16/16
Epoch 24 loss 0.08734001281466747 valid acc 16/16
Epoch 24 loss 0.20017030105157776 valid acc 16/16
Epoch 24 loss 0.30690185395921227 valid acc 16/16
Epoch 24 loss 0.3584374854712551 valid acc 16/16
Epoch 24 loss 0.29729704469053736 valid acc 15/16
Epoch 24 loss 0.13955663561314932 valid acc 16/16
Epoch 24 loss 0.14648821852140312 valid acc 16/16
Epoch 24 loss 0.13542252753311 valid acc 16/16
Epoch 24 loss 0.4089757718559802 valid acc 16/16
Epoch 24 loss 0.13243507616999 valid acc 16/16
Epoch 24 loss 0.4590371619433878 valid acc 16/16
Epoch 24 loss 1.3668691672805393 valid acc 15/16
Epoch 24 loss 0.2513969203837284 valid acc 16/16
Epoch 24 loss 0.27267365906561436 valid acc 16/16
Epoch 24 loss 0.40161487990721406 valid acc 16/16
Epoch 24 loss 0.5519195533303741 valid acc 15/16
Epoch 24 loss 0.09372809393914262 valid acc 16/16
Epoch 24 loss 0.12963835312244196 valid acc 15/16
Epoch 24 loss 0.17720549321958967 valid acc 15/16
Epoch 24 loss 0.18782130555301663 valid acc 15/16
Epoch 24 loss 0.06002647000534811 valid acc 15/16
Epoch 24 loss 0.540107534284556 valid acc 15/16
Epoch 24 loss 0.1453706394005769 valid acc 15/16
Epoch 24 loss 0.30981890097459697 valid acc 15/16
Epoch 24 loss 0.4134909646376964 valid acc 15/16
Epoch 24 loss 0.2126267411229678 valid acc 15/16
Epoch 24 loss 0.10510935341703195 valid acc 15/16
Epoch 24 loss 0.10096892714073219 valid acc 15/16
Epoch 24 loss 0.2065699600621148 valid acc 16/16
Epoch 24 loss 0.753076264567821 valid acc 16/16
Epoch 24 loss 0.12699936894402925 valid acc 15/16
Epoch 24 loss 0.7000398585142363 valid acc 15/16
Epoch 24 loss 0.17765261682147887 valid acc 15/16
Epoch 24 loss 0.5790314300883311 valid acc 16/16
Epoch 24 loss 0.05726957759196437 valid acc 16/16
Epoch 24 loss 0.16506881893098244 valid acc 16/16
Epoch 24 loss 0.22721115279660511 valid acc 16/16
Epoch 24 loss 0.31074095690072473 valid acc 16/16
Epoch 24 loss 0.10822188328169045 valid acc 16/16
Epoch 24 loss 0.2483888602508219 valid acc 16/16
Epoch 25 loss 0.0035901187133385673 valid acc 16/16
Epoch 25 loss 0.7230612998591225 valid acc 16/16
Epoch 25 loss 0.6825851077927394 valid acc 16/16
Epoch 25 loss 0.5365112356902559 valid acc 15/16
Epoch 25 loss 0.5349384323914992 valid acc 15/16
Epoch 25 loss 0.15859466468687405 valid acc 16/16
Epoch 25 loss 0.6012704871824789 valid acc 14/16
Epoch 25 loss 0.4137765344591927 valid acc 15/16
Epoch 25 loss 1.181022590313304 valid acc 16/16
Epoch 25 loss 0.16723999306727477 valid acc 16/16
Epoch 25 loss 0.25556221212802166 valid acc 16/16
Epoch 25 loss 1.3258039439852107 valid acc 16/16
Epoch 25 loss 0.2714754153207399 valid acc 16/16
Epoch 25 loss 0.41518501755875015 valid acc 15/16
Epoch 25 loss 0.23428237615502473 valid acc 15/16
Epoch 25 loss 0.2515046596741189 valid acc 15/16
Epoch 25 loss 0.5111216839330756 valid acc 16/16
Epoch 25 loss 0.43088577614964174 valid acc 16/16
Epoch 25 loss 0.35327306158697547 valid acc 16/16
Epoch 25 loss 0.1273598690997372 valid acc 16/16
Epoch 25 loss 0.5746285561185722 valid acc 15/16
Epoch 25 loss 0.10758350426479985 valid acc 16/16
Epoch 25 loss 0.12452971493237873 valid acc 16/16
Epoch 25 loss 0.09132143673705391 valid acc 16/16
Epoch 25 loss 0.49552107634503906 valid acc 16/16
Epoch 25 loss 0.4680211724268643 valid acc 16/16
Epoch 25 loss 0.2259946077245503 valid acc 16/16
Epoch 25 loss 0.19768360722165004 valid acc 16/16
Epoch 25 loss 0.33414631068936285 valid acc 16/16
Epoch 25 loss 0.08109238101255548 valid acc 16/16
Epoch 25 loss 0.062158833723983356 valid acc 16/16
Epoch 25 loss 0.5844185656411028 valid acc 16/16
Epoch 25 loss 0.2032356855653404 valid acc 16/16
Epoch 25 loss 0.22999677997200713 valid acc 16/16
Epoch 25 loss 1.4384739014491967 valid acc 15/16
Epoch 25 loss 0.43237798713066344 valid acc 16/16
Epoch 25 loss 0.06811559328037077 valid acc 16/16
Epoch 25 loss 0.14081234671391393 valid acc 16/16
Epoch 25 loss 0.1595500273798244 valid acc 15/16
Epoch 25 loss 0.13695797295408205 valid acc 15/16
Epoch 25 loss 0.0637998721696834 valid acc 15/16
Epoch 25 loss 0.295719615967928 valid acc 16/16
Epoch 25 loss 0.21466507199030327 valid acc 15/16
Epoch 25 loss 0.16515797797172893 valid acc 15/16
Epoch 25 loss 0.7243073234591482 valid acc 15/16
Epoch 25 loss 0.18182751581805845 valid acc 15/16
Epoch 25 loss 0.7606812868795301 valid acc 16/16
Epoch 25 loss 0.37797738897912436 valid acc 16/16
Epoch 25 loss 0.32841203342483744 valid acc 16/16
Epoch 25 loss 0.10581822036316375 valid acc 16/16
Epoch 25 loss 0.15653649116593088 valid acc 15/16
Epoch 25 loss 0.226843308005809 valid acc 16/16
Epoch 25 loss 0.40220934027544974 valid acc 15/16
Epoch 25 loss 0.10548328494203221 valid acc 15/16
Epoch 25 loss 0.30222282570232223 valid acc 15/16
Epoch 25 loss 0.3921942482352449 valid acc 15/16
Epoch 25 loss 0.8653738347628162 valid acc 15/16
Epoch 25 loss 0.017172150897271664 valid acc 15/16
Epoch 25 loss 0.31294158262000726 valid acc 15/16
Epoch 25 loss 0.3307671293614357 valid acc 16/16
Epoch 25 loss 0.3306094451298944 valid acc 15/16
Epoch 25 loss 0.09896403587919719 valid acc 15/16
Epoch 25 loss 0.4534281808196867 valid acc 15/16
Epoch 26 loss 0.0004509752007047152 valid acc 15/16
Epoch 26 loss 0.9285102301176993 valid acc 15/16
Epoch 26 loss 0.29859110951850854 valid acc 15/16
Epoch 26 loss 0.29544969857069403 valid acc 14/16
Epoch 26 loss 0.17724982412102738 valid acc 15/16
Epoch 26 loss 0.3861509248506161 valid acc 16/16
Epoch 26 loss 0.4963578234062107 valid acc 16/16
Epoch 26 loss 0.8660195704329439 valid acc 16/16
Epoch 26 loss 0.26054310899048544 valid acc 16/16
Epoch 26 loss 0.30364161582452515 valid acc 16/16
Epoch 26 loss 0.08503640163948094 valid acc 16/16
Epoch 26 loss 0.7265396137577824 valid acc 16/16
Epoch 26 loss 0.2866948070600849 valid acc 14/16
Epoch 26 loss 0.5334932904297115 valid acc 15/16
Epoch 26 loss 0.3113998953097876 valid acc 15/16
Epoch 26 loss 0.272019654210977 valid acc 16/16
Epoch 26 loss 0.48727796873325746 valid acc 16/16
Epoch 26 loss 0.3345261175053088 valid acc 16/16
Epoch 26 loss 0.5902340020458585 valid acc 16/16
Epoch 26 loss 0.12924364247891523 valid acc 16/16

=========
SENTIMENT

Epoch 1, loss 31.368528919209584, train accuracy: 52.44%
Validation accuracy: 52.00%
Best Valid accuracy: 52.00%
Epoch 2, loss 31.180823611871617, train accuracy: 53.78%
Validation accuracy: 53.00%
Best Valid accuracy: 53.00%
Epoch 3, loss 31.093596430061663, train accuracy: 54.22%
Validation accuracy: 60.00%
Best Valid accuracy: 60.00%
Epoch 4, loss 31.070355638720848, train accuracy: 56.67%
Validation accuracy: 61.00%
Best Valid accuracy: 61.00%
Epoch 5, loss 30.730947217937505, train accuracy: 57.78%
Validation accuracy: 66.00%
Best Valid accuracy: 66.00%
Epoch 6, loss 30.77625198177194, train accuracy: 59.33%
Validation accuracy: 63.00%
Best Valid accuracy: 66.00%
Epoch 7, loss 30.247528268002245, train accuracy: 62.22%
Validation accuracy: 61.00%
Best Valid accuracy: 66.00%
Epoch 8, loss 30.03459122945942, train accuracy: 62.89%
Validation accuracy: 66.00%
Best Valid accuracy: 66.00%
Epoch 9, loss 29.589783068712443, train accuracy: 66.44%
Validation accuracy: 64.00%
Best Valid accuracy: 66.00%
Epoch 10, loss 29.06925665707594, train accuracy: 68.00%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 11, loss 29.41759935400795, train accuracy: 66.67%
Validation accuracy: 66.00%
Best Valid accuracy: 68.00%
Epoch 12, loss 28.911767061667636, train accuracy: 68.89%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 13, loss 28.2651659192271, train accuracy: 70.67%
Validation accuracy: 67.00%
Best Valid accuracy: 71.00%
Epoch 14, loss 27.74103717349798, train accuracy: 72.00%
Validation accuracy: 70.00%
Best Valid accuracy: 71.00%
Epoch 15, loss 27.883648974031576, train accuracy: 70.89%
Validation accuracy: 69.00%
Best Valid accuracy: 71.00%
Epoch 16, loss 26.691848261195258, train accuracy: 75.11%
Validation accuracy: 70.00%
Best Valid accuracy: 71.00%
Epoch 17, loss 26.507727682887428, train accuracy: 75.11%
Validation accuracy: 69.00%
Best Valid accuracy: 71.00%
Epoch 18, loss 26.004426035191383, train accuracy: 74.22%
Validation accuracy: 70.00%
Best Valid accuracy: 71.00%
Epoch 19, loss 24.52078628740038, train accuracy: 76.44%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 20, loss 25.150752817064287, train accuracy: 75.56%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 21, loss 23.7982700711223, train accuracy: 76.89%
Validation accuracy: 71.00%
Best Valid accuracy: 73.00%
Epoch 22, loss 23.292582933828008, train accuracy: 78.67%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 23, loss 22.876417556843396, train accuracy: 77.56%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 24, loss 22.234050135476398, train accuracy: 81.11%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 25, loss 21.240766510464784, train accuracy: 82.22%
Validation accuracy: 76.00%
Best Valid accuracy: 76.00%
Epoch 26, loss 20.648642476394578, train accuracy: 81.78%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 27, loss 20.317931064276383, train accuracy: 82.44%
Validation accuracy: 76.00%
Best Valid accuracy: 76.00%
Epoch 28, loss 19.120950167094218, train accuracy: 84.00%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 29, loss 18.5813842891722, train accuracy: 84.44%
Validation accuracy: 77.00%
Best Valid accuracy: 77.00%
Epoch 30, loss 18.632714663989038, train accuracy: 84.22%
Validation accuracy: 78.00%
Best Valid accuracy: 78.00%
Epoch 31, loss 17.397824408522926, train accuracy: 84.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 32, loss 17.759725905931884, train accuracy: 83.78%
Validation accuracy: 78.00%
Best Valid accuracy: 78.00%
Epoch 33, loss 16.601606105820327, train accuracy: 84.89%
Validation accuracy: 78.00%
Best Valid accuracy: 78.00%
Epoch 34, loss 16.52098658021734, train accuracy: 86.00%
Validation accuracy: 76.00%
Best Valid accuracy: 78.00%
Epoch 35, loss 14.952527407061268, train accuracy: 89.11%
Validation accuracy: 76.00%
Best Valid accuracy: 78.00%
Epoch 36, loss 15.233734255613141, train accuracy: 87.33%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%
Epoch 37, loss 14.149893126367383, train accuracy: 88.22%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 38, loss 13.37974737775393, train accuracy: 91.11%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 39, loss 14.043923403912691, train accuracy: 90.00%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 40, loss 13.25547496718058, train accuracy: 91.33%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 41, loss 12.63722282078611, train accuracy: 91.56%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 42, loss 12.895354681674627, train accuracy: 90.44%
Validation accuracy: 78.00%
Best Valid accuracy: 79.00%
Epoch 43, loss 12.232229456557398, train accuracy: 92.89%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 44, loss 11.36668680355454, train accuracy: 93.78%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 45, loss 12.068925609924186, train accuracy: 91.33%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 46, loss 11.328255703263586, train accuracy: 91.78%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 47, loss 11.072314044300514, train accuracy: 92.89%
Validation accuracy: 77.00%
Best Valid accuracy: 79.00%
Epoch 48, loss 10.525179768013055, train accuracy: 93.33%
Validation accuracy: 76.00%
Best Valid accuracy: 79.00%
Epoch 49, loss 9.823589185402305, train accuracy: 95.33%
Validation accuracy: 74.00%
Best Valid accuracy: 79.00%
Epoch 50, loss 9.725028727175959, train accuracy: 94.00%
Validation accuracy: 79.00%
Best Valid accuracy: 79.00%