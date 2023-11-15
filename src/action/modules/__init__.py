from action.modules.module import ClassifierModule, ClassifierSeqModule, ClassifierSeqModuleBS
from action.modules.segmenter import SegmenterModule

all_modules = {
  "cls": ClassifierModule,  # multi-class classifier module
  "cls_seq": ClassifierSeqModule,  # multi-class classifier module applied to sequential data
  "cls_seq_bsoftmax": ClassifierSeqModuleBS, # multi-class classifier module applied to sequential data using balanced softmax loss
  "segmenter_module": SegmenterModule,
}