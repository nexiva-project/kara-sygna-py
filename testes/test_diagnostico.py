"""Testes do arquivo de diagnóstico do reconhecimento."""

from kara_sygna.aplicacao.diagnostico import DiagnosticoReconhecimento


def test_duas_sessoes_no_mesmo_segundo_nao_se_sobrescrevem(tmp_path):
    primeira = DiagnosticoReconhecimento(tmp_path)
    segunda = DiagnosticoReconhecimento(tmp_path)
    primeira.registrar({"motivo": "ok", "escrito": "casa"})
    assert primeira.caminho != segunda.caminho
    primeira.fechar()
    segunda.fechar()
    assert "casa" in primeira.caminho.read_text(encoding="utf-8")
