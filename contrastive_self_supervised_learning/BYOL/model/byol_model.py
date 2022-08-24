#-*- coding:utf-8 -*-
import torch
from .basic_modules import EncoderwithProjection, Predictor

class BYOLModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # online network
        self.online_network = EncoderwithProjection(config)

        # target network
        self.target_network = EncoderwithProjection(config)

        # predictor
        self.predictor = Predictor(config)

        self._initializes_target_network()

    @torch.no_grad()
    def _initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

    @torch.no_grad()
    def _update_target_network(self, mm):
        """Momentum update of target network"""
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.mul_(mm).add_(1. - mm, param_q.data)

    def forward(self, view1, view2, mm):
        # online network forward
        # if torch.rand(1).round():
        #     q = self.predictor(self.online_network(view1))
        #      # target network forward
        #     with torch.no_grad():
        #         self._update_target_network(mm)
        #         target_z = self.target_network(view2).detach().clone()
        # else:
        #     q = self.predictor(self.online_network(view2))
        #      # target network forward
        #     with torch.no_grad():
        #         self._update_target_network(mm)
        #         target_z = self.target_network(view1).detach().clone()
        q1 = self.predictor(self.online_network(view1))
        q2 = self.predictor(self.online_network(view2))
        with torch.no_grad():
                self._update_target_network(mm)
                target_z1 = self.target_network(view2).detach().clone()
                target_z2 = self.target_network(view1).detach().clone()
        return q1,q2,target_z1, target_z2
