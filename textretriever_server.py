import time
import grpc
from concurrent import futures
import argparse
import torch
from transformers import *
from rank_bm25 import BM25Okapi
from textretriever import OperationStatus, TextRetrievalResponse, RetrievalResult, ListingResponse, Candidate, OperationResponse, RetrieverStatus, DeletionRequest, AddOrUpdateRequest, ListingRequest
from textretriever_grpc import TextRetrieverServicer, TextRetrieverServicer_to_server
import numpy as np
import pandas as pd
import os

class TextRetrieverServer(TextRetrieverServicer):
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ### dual encoder ###
        self.q_tokenizer = AutoTokenizer.from_pretrained("KoBigBird")
        self.q_encoder = AutoModel.from_pretrained("KoBigBird") # query encoder
        self.q_encoder.to(self.device)
        self.p_tokenizer = AutoTokenizer.from_pretrained("KoBigBird")
        self.p_encoder = AutoModel.from_pretrained("KoBigBird") # passage encoder
        self.p_encoder.to(self.device)
        self.dual_passage_embeddings = torch.load("dual_candidate_embs.pt")
        self.dual_candidate_df = pd.read_csv("dual_candidate_texts.csv")
        self.dual_candidate_texts = self.dual_candidate_df["passages"].values
        # create list of candidate objects
        self.dual_candidate_info = []
        for i in range(len(self.dual_candidate_texts)):
            cur_candidate = Candidate(text = self.dual_candidate_texts[i], id = str(i), appendix = {"method":"DPR"})
            self.dual_candidate_info.append(cur_candidate)

        ### mono encoder ###
        self.mono_tokenizer = AutoTokenizer.from_pretrained("KoBigBird")
        self.mono_encoder = AutoModel.from_pretrained("KoBigBird")
        self.mono_encoder.to(self.device)
        self.mono_passage_embeddings = torch.load("mono_candidate_embs.pt")
        self.mono_candidate_df = pd.read_csv("mono_candidate_texts.csv")
        self.mono_candidate_texts = self.mono_candidate_df["passages"].values
        # create list of candidate objects
        self.mono_candidate_info = []
        for i in range(len(self.mono_candidate_texts)):
            cur_candidate = Candidate(text = self.mono_candidate_texts[i], id = str(i), appendix = {"method":"MONO"})
            self.mono_candidate_info.append(cur_candidate)

        ### BM25 ###
        ### we create tokenized corpus using kobigbird tokenizer ###
        self.bm25_tokenizer = AutoTokenizer.from_pretrained("KoBigBird")
        self.bm25_candidate_df = pd.read_csv("bm25_candidate_texts.csv")
        self.bm25_candidate_texts = self.bm25_candidate_df["passages"].values
        self.bm25_tokenized_corpus = [self.bm25_tokenizer.tokenize(doc) for doc in self.bm25_candidate_texts] # this can also be prepared like the embeddings
        self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)
        # create list of candidate objects
        self.bm25_candidate_info = []
        for i in range(len(self.candidate_texts)):
            cur_candidate = Candidate(text = self.bm25_candidate_texts[i], id = str(i), appendix = {"method":"BM25"})
            self.bm25_candidate_info.append(cur_candidate)

    def RetrieveTexts(self, request, context):
        self.query_texts = request.query_texts
        self.num_candidates = request.num_candidates
        with torch.no_grad():
            try:
                ### DPR ###
                self.q_encoder.eval()
                self.p_encoder.eval()
                retrieval_results = []
                # get ranked candidate lists for each query
                for i in range(len(self.query_texts)):
                    encoded_query = self.q_tokenizer(str(self.query_texts[i]), max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(self.device)
                    q_emb = self.q_encoder(**encoded_query).pooler_output
                    dot_prod_scores = torch.matmul(self.q_emb, torch.transpose(self.dual_passage_embeddings, 0, 1))
                    ranks = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
                    scores = torch.sort(dot_prod_scores, dim=1, descending=True).squeeze()
                    ranks = ranks[:self.num_candidates]
                    scores = scores[:self.num_candidates]
                    dual_retrieval_result = []
                    for j in range(len(ranks)):
                        score = scores[j]
                        self.dual_candidate_info[ranks[j]].score = score
                        dual_retrieval_result.append(self.dual_candidate_info[ranks[j]])
                    retrieval_results.append(RetrievalResult(query_text=self.query_texts[i], candidates=dual_retrieval_result))

                ### MONO ###
                self.mono_encoder.eval()
                for i in range(len(self.query_texts)):
                    encoded_query = self.mono_tokenizer(str(self.query_texts[i]), max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(self.device)
                    q_emb = self.mono_encoder(**encoded_query).pooler_output
                    q_emb = q_emb.detach().cpu().numpy()
                    sim_scores = cdist(q_emb, self.mono_passage_embeddings, "cosine")[0]
                    ranks = np.argsort(sim_scores)
                    scores = np.sort(sim_scores)
                    ranks = ranks[:self.num_candidates]
                    scores = scores[:self.num_candidates]
                    mono_retrieval_result = []
                    for j in range(len(ranks)):
                        score = scores[j]
                        self.mono_candidate_info[ranks[j]].score = score
                        mono_retrieval_result.append(self.mono_candidate_info[ranks[j]])
                    retrieval_results.append(RetrievalResult(query_text=self.query_texts[i], candidates=mono_retrieval_result))

                ### BM25 ###
                for i in range(len(self.query_texts)):
                    tok_query = self.bm25_tokenizer.tokenize(self.query_texts[i])
                    doc_scores = self.bm25.get_scores(tok_query)
                    ranks = np.argsort(doc_scores)[::-1]
                    scores = np.sort(doc_scores)[::-1]
                    ranks = ranks[:self.num_candidates]
                    scores = scores[:self.num_candidates]
                    bm25_retrieval_result = []
                    for j in range(len(ranks)):
                        score = scores[j]
                        self.bm25_candidate_info[ranks[j]].score = score
                        bm25_retrieval_result.append(self.bm25_candidate_info[ranks[j]])
                    retrieval_results.append(RetrievalResult(query_text=self.query_texts[i], candidates=bm25_retrieval_result))
                return TextRetrievalResponse(status = SUCCESS, results = retrieval_results)
            except:
                return TextRetrievalResponse(status = FAILURE, results = [])

    def ListCandidates(self, request, context):
        self.begin_id = request.begin_id
        self.count_list = request.count
        assert self.begin_id >= 0
        assert self.count_list >= 1
        ret_list = self.dual_candidate_info[self.begin_id:self.begin_id + self.count_list]
        return ListingResponse(status=SUCCESS, candidates=ret_list)

    def AddOrUpdateCandidates(self, request, context):
        self.to_be_added = self.candidates
        cnt = 0
        for i in range(len(self.to_be_added)):
            try:
                new_cand = self.to_be_added[i]
                new_cand.id = str(len(self.dual_candidate_info) + i + 1) # assign id position and append it to the list
                self.dual_candidate_info.append(new_cand)
                text = new_cand.text
                with torch.no_grad():
                    encoded_p = self.p_tokenizer(str(text), max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(self.device)
                    p_emb = self.p_encoder(**encoded_p).pooler_output
                    p_emb = p_emb.detach().cpu()
                    self.dual_passage_embeddings = torch.cat((self.dual_passage_embeddings, p_emb), dim=0)

                self.mono_candidate_info.append(new_cand)
                with torch.no_grad():
                    encoded_p = self.mono_tokenizer(str(text), max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(self.device)
                    p_emb = self.mono_encoder(**encoded_p).pooler_output
                    p_emb = p_emb.detach().cpu().numpy()
                    self.mono_passage_embeddings = np.concatenate([self.mono_passage_embeddings, p_emb], axis=0)

                self.bm25_candidate_info.append(new_cand)
                self.bm25_tokenized_corpus.append(self.bm25_tokenizer.tokenize(text))
                cnt += 1
            except Exception as e:
                print(e)
                continue
        return OperationResponse(status = SUCCESS, success_count = cnt)

    def DeleteCandidates(self, request, context):
        self.to_be_deleted = self.ids
        cnt = 0
        for i in range(len(self.to_be_deleted)):
            try:
                for j in range(len(self.dual_candidate_info)):
                    if dual_candidate_info[j].id == self.to_be_deleted[i]:
                        dual_candidate_info.pop(j)
                    if mono_candidate_info[j].id == self.to_be_deleted[i]:
                        mono_candidate_info.pop(j)
                    if bm25_candidate_info[j].id == self.to_be_deleted[i]:
                        bm25_candidate_info.pop(j)
                cnt += 1
            except Exception as e:
                print(e)
                continue
        return OperationResponse(status = SUCCESS, success_count = cnt)

    def CheckStatus(self, request, context):
        num_total_candidates = len(self.dual_candidate_info)
        appendix_keys = list(self.dual_candidate_info[0].appendix.keys())
        return RetrieverStatus(num_total_candidates = num_total_candidates, appendix_keys = appendix_keys)
