from settings import *
from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertTokenizerFast
import torch


from question_answering import QAModel

path = '../models/fquad_distilbert_qa'
tokenizer = DistilBertTokenizerFast.from_pretrained(path, local_files_only=True)
#model =  DistilBertForQuestionAnswering.from_pretrained(path)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = QAModel().to(device)

model.load_state_dict(torch.load(os.path.join(MODELS_DIR,'custom_model_fquad_distilbert.pth')))


question ="Quel est le nom du panneau central du retable ?"
context = "Les chercheurs s'accordent pour considérer que les deux panneaux sont issus de l'atelier de Léonard de Vinci : en effet, ils se rapprochent du style que le maître déploie dans la version londonienne du panneau central du retable (La Vierge aux rochers) ; par ailleurs, les trois panneaux du retable témoignent de procédés techniques similaires — utilisation des doigts par exemple pour la finition des contours des figures — qui, pour certains, sont à l'origine de problèmes identiques — telles des craquelures de la couche picturale apparaissant dès le processus de séchage. "



def predict(question, context,model, tokenizer):
    inputs = tokenizer(question, context, return_tensors="pt").to(device)

    with torch.no_grad():
        start_logits, end_logits = model(**inputs)

    answer_start_index = start_logits.argmax()
    answer_end_index = end_logits.argmax()

    print(answer_start_index, answer_end_index)

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

    return tokenizer.decode(predict_answer_tokens)

print(predict(question,context,model,tokenizer))
