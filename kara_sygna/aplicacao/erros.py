"""Erros previstos da aplicação, com mensagens prontas para mostrar ao usuário."""

from __future__ import annotations


class ErroAplicacao(Exception):
    """Base dos erros "esperados": a mensagem já é amigável e pode ir direto para a tela."""


class PalavraInvalidaError(ErroAplicacao):
    """A palavra do sinal está vazia."""


class SemDadosError(ErroAplicacao):
    """Não há exemplos gravados."""


class ModeloAusenteError(ErroAplicacao):
    """A IA ainda não foi treinada."""


class ModeloDesatualizadoError(ErroAplicacao):
    """Os sinais mudaram depois do último treino."""


class DuracaoInvalidaError(ErroAplicacao):
    """O tempo de gravação escrito não é um número de segundos válido."""
