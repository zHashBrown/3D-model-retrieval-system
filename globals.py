"""
constants for the data set.
ModelNet40 for example
"""
NUM_CLASSES = 40
NUM_VIEWS = 12
TRAIN_LOL = './data/view/train_lists.txt'
VAL_LOL = './data/view/val_lists.txt'
TEST_LOL = './data/view/test_lists.txt'
ALL_LOL = './data/view/all_lists.txt'


"""
constants for both training and testing
"""
BATCH_SIZE = 16

# this must be more than twice the BATCH_SIZE
INPUT_QUEUE_SIZE = 4 * BATCH_SIZE


"""
constants for training the model
"""
INIT_LEARNING_RATE = 0.0001

# sample how many shapes for validation
# this affects the validation time
VAL_SAMPLE_SIZE = 256

# do a validation every VAL_PERIOD iterations
# 这里由于我内存吃不消，所以改了非常大的数来默认不验证，要做测试请暂停后执行test.py
VAL_PERIOD = 1000000

# save the progress to checkpoint file every SAVE_PERIOD iterations
# this takes tens of seconds. Don't set it smaller than 100.
SAVE_PERIOD = 1000

