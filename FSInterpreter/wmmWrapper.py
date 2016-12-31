import ctypes 
from numpy.ctypeslib import ndpointer

lib = ctypes.cdll.LoadLibrary('./lib/libwordmodelmgr.so')

wmm_add_letter_prediction = lib.WordModelMgr_AddLetterPrediction
wmm_add_letter_prediction.restype = None
wmm_add_letter_prediction.argtypes = [ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_double,
                                      ctypes.c_double]

wmm_get_best_prediction = lib.WordModelMgr_GetBestPrediction
wmm_get_best_prediction.restype = ctypes.c_char_p
wmm_get_best_prediction.argtypes = [ctypes.c_void_p]

wmm_get_next_prediction = lib.WordModelMgr_GetNextPrediction
wmm_get_next_prediction.restype = ctypes.c_char_p
wmm_get_next_prediction.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]


class WordModelMgr(object):
    def __init__(self):
        self.obj = lib.WordModelMgr_new()

    def __del__(self):
        lib.WordModelMgr_delete(self.obj)

    def initialize(self):
        return lib.WordModelMgr_Initialize(self.obj)

    def add_letter_prediction(self, letter_index, confidence, double_prob):
        wmm_add_letter_prediction(self.obj, letter_index, confidence, double_prob)

    def finalize_prediction(self):
        lib.WordModelMgr_FinalizePrediction(self.obj)

    def get_best_prediction(self):
        return wmm_get_best_prediction(self.obj)

    def get_next_prediction(self, prob):
        return wmm_get_next_prediction(self.obj, prob)

    def reset(self):
        lib.WordModelMgr_Reset(self.obj)

    def dump_candidates(self):
        lib.WordModelMgr_DumpCandidates(self.obj)
