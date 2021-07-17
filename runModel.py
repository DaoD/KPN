import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import logging
from torch.utils import data
from tqdm import tqdm
from kpn import KPN
from evaluation import evaluate_list

dataset = "duconv"
# dataset = "durecdial"

if dataset == "duconv":
    embedding_file = "./data/duconv/embeddings.pkl"
    train_data_file = "./data/duconv/train.pt"
    val_data_file = "./data/duconv/dev.pt"
    test_data_file = "./data/duconv/test.pt"
    result_log_file = "./output/duconv/output_log"
    train_model_path = "./output/duconv/model/"
else:
    embedding_file = "../data/DuRecDial/embeddings.pkl"
    train_data_file = "../data/DuRecDial/train.pt"
    val_data_file = "../data/DuRecDial/dev.pt"
    test_data_file = "../data/DuRecDial/test.pt"
    result_log_file = "../output/DuRecDial/output_log"
    train_model_path = "../output/DuRecDial/model/"

logging.basicConfig(filename=result_log_file, level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0")

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Dataset(data.TensorDataset):
    def __init__(self, context, response, knowledge, goal, knowledge_mask, y_true):
        super(Dataset, self).__init__()
        self.context = torch.LongTensor(context)
        self.response = torch.LongTensor(response)
        self.knowledge = torch.LongTensor(knowledge)
        self.goal = torch.LongTensor(goal)
        self.knowledge_mask = torch.LongTensor(knowledge_mask)
        self.y_true = torch.LongTensor(y_true)

    def __len__(self):
        return self.y_true.size(0)

    def __getitem__(self, index):
        return self.context[index], self.response[index], self.knowledge[index], self.goal[index], self.knowledge_mask[index], self.y_true[index]

def train(model_path=None, Lambda=None):
    set_seed()
    batch_size = 128
    max_epoch = 5
    decay_rate = 0.5

    if model_path:
        if not os.path.exists(model_path):
            os.mkdir(model_path)
    with open(embedding_file, "rb") as f:
        embeddings = pickle.load(f, encoding="bytes")
        embeddings = torch.FloatTensor(embeddings)
    train, test = torch.load(train_data_file), torch.load(val_data_file)

    context_train = train['c']
    response_train = train['r']
    knowledge_train = train['k']
    goal_train = train['g']
    knowledge_mask_train = train['kn_mask']
    y_train = train['y']

    context_test = test['c']
    response_test = test['r']
    knowledge_test = test['k']
    goal_test = test['g']
    knowledge_mask_test = test['kn_mask']
    y_test = test['y']

    model = KPN(dataset=dataset, embedding=embeddings, device=device)
    model = model.cuda()

    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('* number of parameters: %d' % n_params)

    loss_func = nn.BCEWithLogitsLoss()
    loss_for_knowledge = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-8)

    training_set = Dataset(context_train, response_train, knowledge_train, goal_train, knowledge_mask_train, y_train)
    training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

    test_set = Dataset(context_test, response_test, knowledge_test, goal_test, knowledge_mask_test, y_test)
    test_generator = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    save_step = 500
    best_result = [0.0, 0.0]
    patience = 0
    logger.info("Start Lambda" + str(Lambda))
    for epoch in range(max_epoch):
        print("\nEpoch ", epoch + 1, "/", max_epoch)
        epoch_step = 0
        step = 0
        train_loss = 0.0
        with tqdm(total=len(y_train), ncols=100) as pbar:
            for batch in training_generator:
                batch = tuple(t.to(device) for t in batch)
                context, response, knowledge, goal, knowledge_mask, y_true = batch
                optimizer.zero_grad()
                knowledge_mask = knowledge_mask.float()
                logits, knowledge_selector = model(context, response, knowledge, goal, knowledge_mask)
                y_true = y_true.float()
                loss = loss_func(logits, y_true)
                loss_knowledge = loss_for_knowledge(knowledge_selector, knowledge_mask)
                loss = loss + Lambda * loss_knowledge
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5)
                optimizer.step()
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                pbar.set_postfix(lr=current_lr, loss=loss.item(), _l=Lambda)
                pbar.update(batch_size)
                train_loss += loss.item()
                step += 1
                epoch_step += 1
                if epoch_step % save_step == 0:
                    model.eval()
                    all_candidate_scores = []
                    val_loss = 0.0
                    val_step = 0
                    with torch.no_grad():
                        for test_batch in test_generator:
                            test_batch = tuple(t.to(device) for t in test_batch)
                            context_test, response_test, knowledge_test, goal_test, knowledge_mask_test, y_true_test = test_batch
                            knowledge_mask_test = knowledge_mask_test.float()
                            y_true_test = y_true_test.float()
                            logits_test, knowledge_selector_test = model(context_test, response_test, knowledge_test, goal_test, knowledge_mask_test)
                            loss_test = loss_func(logits_test, y_true_test)
                            loss_knowledge_test = loss_for_knowledge(knowledge_selector_test, knowledge_mask_test)
                            loss_test = loss_test + Lambda * loss_knowledge_test
                            val_step += 1
                            val_loss += loss_test.item()
                            all_candidate_scores.append(logits_test.data.cpu().numpy())
                        all_candidate_scores = np.concatenate(all_candidate_scores, axis=0).tolist()
                        result = evaluate_list(all_candidate_scores, y_test, 10)
                        if result[0] + result[1] > best_result[0] + best_result[1]:
                            logger.info("lambda - %.2f, Hits@1: %.4f, Hits@3: %.4f, MRR: %.4f" % (Lambda, result[0], result[1], result[2]))
                            tqdm.write("Hits@1: %.4f, Hits@3: %.4f, MRR: %.4f" % (result[0], result[1], result[2]))
                            torch.save(model.state_dict(), model_path + "model.pth")
                            best_result = result
                            patience = 0
                        else:
                            patience += 1
                    step = 0
                    train_loss = 0.0
                    model.train()
                if epoch >= 1 and patience >= 3:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * decay_rate
                    model.load_state_dict(torch.load(model_path + "model.pth"))
                    patience = 0
    logger.info("lambda - %.2f, Best Hits@1: %.4f, Hits@3: %.4f, MRR: %.4f" % (Lambda, best_result[0], best_result[1], best_result[2]))
    tqdm.write("Finish lambda - %.2f, Best Hits@1: %.4f, Hits@3: %.4f, MRR: %.4f" % (Lambda, best_result[0], best_result[1], best_result[2]))

def test(model_path, test_data_file, ground_truth_path, result_path, save_kn=False, save_kn_path=None):
    from EvaluationUtils import write_result
    set_seed()
    batch_size = 128
    with open(embedding_file, "rb") as f:
        embeddings = pickle.load(f, encoding="bytes")
        embeddings = torch.FloatTensor(embeddings)
    test_data = torch.load(test_data_file)

    context_test = test_data['c']
    response_test = test_data['r']
    knowledge_test = test_data['k']
    goal_test = test_data['g']
    knowledge_mask_test = test_data['kn_mask']
    y_test = test_data['y']

    test_set = Dataset(context_test, response_test, knowledge_test, goal_test, [0] * len(context_test), y_test)
    test_generator = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = KPN(embedding=embeddings, device=device)
    model = model.to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    all_candidate_scores = []
    all_kn_scores = []
    with torch.no_grad():
        with tqdm(total=len(y_test)) as pbar:
            for batch in test_generator:
                batch = tuple(t.to(device) for t in batch)
                context_test, response_test, knowledge_test, goal_test, _, y_true_test = batch
                y_true_test = y_true_test.float()
                logits_test, knowledge_selector_test = model(context_test, response_test, knowledge_test, goal_test)
                all_candidate_scores.append(logits_test.data.cpu().numpy())
                all_kn_scores.append(knowledge_selector_test.data.cpu().numpy())
                pbar.update(batch_size)
        all_candidate_scores = np.concatenate(all_candidate_scores, axis=0).tolist()
        result = evaluate_list(all_candidate_scores, y_test, negtive_sample=10)
        print("Hits@1: %f, Hits@3: %f, MRR: %f" % (result[0], result[1], result[2]))

train(train_model_path, 0.3)