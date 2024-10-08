import editdistance

# Based on seminar materials
# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    # DONE
    if len(target_text) == 0:
        return 1.0
    
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(target_text.split())

def calc_wer(target_text, predicted_text) -> float:
    # DONE
    if len(target_text) == 0:
        return 1.0
    
    return editdistance.eval(target_text, predicted_text) / len(target_text.split())
