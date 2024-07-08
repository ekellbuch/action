from action.modules.module import ClassifierModule, ClassifierSeqModule, ClassifierSeqModuleBS
from action.modules.segmenter import SegmenterModule, SegmenterBsoftModule, SegmenterBsoftWeakModule, SegmenterBsoftSWeakModule
from action.modules.baseline import RegressionNOBPModule


all_modules = {
  "cls": ClassifierModule,  # multi-class classifier module
  "cls_seq": ClassifierSeqModule,  # multi-class classifier module applied to sequential data
  "cls_seq_bsoftmax": ClassifierSeqModuleBS, # multi-class classifier module applied to sequential data using balanced softmax loss
  "segmenter_module": SegmenterModule,  # multi-class classifier module
  "segmenterBsoft_module": SegmenterBsoftModule,  # apply balanced softmax to strong labels
  "segmenterBsoftWeak_module": SegmenterBsoftWeakModule,  # apply balanced softmax to weak labels weighted by strong labels
  "segmenterBsoftSWeak_module": SegmenterBsoftSWeakModule, # apply balanced softmax to weak labels weighted by weak labels
  "regression_nobp": RegressionNOBPModule,  # regression module
}