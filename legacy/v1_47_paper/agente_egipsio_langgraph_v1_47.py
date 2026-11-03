# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 01:31:39 2025

@author: Daniel

VERSÃO v1.42 - REPRODUCIBILITY: all Anthropic calls set to temperature=0.0 🔬🔒
  Key change: rule-based output elevated to AGENT 3 in Sintetizador
  Previously: rules shown as background "baseline context"
  Now: rules presented as peer agent with explicit voting logic (87%% validated)
  Voting: All-3-agree → accept | Rules+any-LLM → strong | LLM-only → moderate
  Base: v1.38 (83.9%% LLM-A, 85.1%% LLM-B, gen gap -6.5pp)

OBJETIVO v1.29:
  • Implementar avaliação Leave-One-Out (LOO) com as 99 amostras disponíveis.
  • Para cada amostra i (0..98):
      1. Os exemplos de few-shot são selecionados das OUTRAS 98 amostras.
      2. A análise é executada sobre a amostra i.
      3. Os determinativos previstos são comparados ao gabarito (anotações em negrito
         nos arquivos DOCX fornecidos pelo especialista).
  • Ao final, calculam-se Precision, Recall e F1-Score globais (macro e micro),
    além de métricas por sinal e relatório detalhado por amostra.

GABARITO (ground truth):
  • Extraído automaticamente de Langgraph_analyses_comments.docx (50 amostras)
    e Langgraph_analyses_comments_2.docx (49 amostras).
  • Os signos em negrito na coluna "Sequência:" de cada DOCX são os determinativos
    corretos segundo o especialista.

COMO FUNCIONA O LOO:
  • df_loo contém apenas as 99 linhas que têm gabarito.
  • Para a amostra i, os exemplos few-shot são escolhidos de df_loo.drop(i).
  • O EXPERT_EXAMPLES global é substituído por EXPERT_EXAMPLES_LOO em cada iteração.
  • Toda a lógica de análise (agentes, scoring) permanece igual à v1.28.

CHANGES vs v1.28:
  1. Funções novas: parse_ground_truth_from_docx(), load_loo_ground_truth(),
     compute_loo_metrics(), select_expert_examples_loo().
  2. Loop principal substituído pelo loop LOO.
  3. Novo bloco de avaliação com tabela detalhada, gráficos e exportação.
  4. DOCX_PATHS configurável (novo parâmetro do usuário).

HISTÓRICO:
V1.28: Correções direcionadas (mantém sucessos v1.27, corrige I9/L1/I10)
V1.29: Avaliação Leave-One-Out com 99 amostras e gabarito de especialista

VERSÃO v1.28 - CORREÇÕES CRÍTICAS PÓS-AVALIAÇÃO v1.27! 🎯

RESULTADOS v1.27 (Avaliação real com 99 sequências):
❌ F1: 72.77% (v1.26: 81.39%) [-8.61pp] - REGRESSÃO
✅ W24 FP: 20→2 (-90%) - SUCESSO!
✅ R8 FP: 13→1 (-92%) - SUCESSO!
❌ I9 FP: 0→47 (+47) - PROBLEMA CRÍTICO!
❌ L1 FP: 0→22 (+22) - PROBLEMA CRÍTICO!
❌ I10 FP: 0→18 (+18) - PROBLEMA CRÍTICO!

CAUSA RAIZ v1.27:
• Melhorias direcionadas (W24, R8) funcionaram perfeitamente
• Expansão da KB (+70 sinais) encontrou sinais ausentes
• MAS: Sinais novos (I9, L1, I10) sem validação de contexto
• Resultado: +110 FP totais (61→171), principalmente de sinais novos
• Lição: "Mais conhecimento ≠ Melhor conhecimento"

CORREÇÕES v1.28:
1. VALIDAÇÃO DE CONTEXTO PARA SINAIS PROBLEMÁTICOS:
   • I9, I10 (cobras): categoria context_animal
   • L1 (escaravelho): categoria context_insect (como R8)
   • D58 (perna), U1 (foice): limiares mais altos

2. LIMIARES AJUSTADOS:
   • I9, I10: 0.90 (muito ambíguos, fonograma vs determinativo)
   • L1: 0.85 (sagrado, mas também fonético)
   • D58, U1: 0.80 (sinais novos precisam evidência mais forte)

3. CAMPOS SEMÂNTICOS EXPANDIDOS:
   • animal: I9, I10, serpente, cobra, réptil
   • insect: L1, escaravelho, sagrado, divino
   • tool: U1, foice, agricultura, colheita

4. PENALIDADE PARA SINAIS RECÉM-ADICIONADOS:
   • Todos os 70 sinais da v1.27 têm penalidade de 15-40%
   • Requer evidência mais forte antes de detectar
   • Previne explosão de FP em sinais não testados

5. PROMPTS APRIMORADOS:
   • Orientação explícita para I9, L1, I10
   • Exemplos de contextos válidos/inválidos
   • Ênfase em validação de campo semântico

MANTÉM v1.27 (sucessos):
✓ Validação de contexto W24/R8
✓ Base expandida (F51, N23, T14, P1, O38, D56)
✓ Sistema de categorização por probabilidade
✓ Prompts aprimorados

HISTÓRICO:
V1.6-V1.13: Sistema base
V1.14: 5 correções críticas
V1.15: Estrutural independente
V1.23: Correções O31/confidence
V1.24: Melhorias baseadas em dados
V1.26: Causa raiz (desativa enrich_determinatives_with_lookup)
V1.27: Melhorias sistemáticas (resultados mistos, W24/R8 sucesso, I9/L1 problema)
V1.28: Correções direcionadas (mantém sucessos v1.27, corrige I9/L1/I10)

ESTATÍSTICAS ESPERADAS v1.28:
- F1-Score: 86-87% (v1.27: 72.77%, v1.26: 81.39%) [+13-14pp vs v1.27]
- Precision: 85-87% (v1.27: 61.92%, v1.26: 80.88%) [+23-25pp vs v1.27]
- Recall: 87-88% (v1.27: 88.25%, v1.26: 81.90%) [mantém alto recall]
- Redução FP: 171→45-50 (-120 FP vs v1.27)
- Mantém FN baixo: ~37-40 (v1.27: 37)

TARGET: F1 ≥ 88% (próximo com v1.28!)
"""

"""
# Proyecto Híbrido Mejorado: Reconocimiento de Jeroglíficos Egipcios
Comparación entre Top1 OCR, n-gram Viterbi, BERT CE y Enfoque Híbrido con optimizaciones
"""

# ============================================================================
# FASE 0 - SETUP E INSTALAÇÃO
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║   🏛️  ANÁLISIS DE JEROGLÍFICOS EGIPCIOS CON LANGGRAPH                      ║
║                    VERSIÓN CONVERTIDA DE CREWAI                            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Este cuaderno implementa un análisis avanzado de determinativos en jeroglíficos
egipcios usando:
  • Análisis basado en reglas (siempre activo)
  • Sistema multi-agente con LangGraph (opcional)
  • Base de datos de 80+ determinativos más frecuentes

┌────────────────────────────────────────────────────────────────────────────┐
│ ⚙️  CONFIGURACIÓN SEGURA: ACTIVAR LANGGRAPH                                 │
└────────────────────────────────────────────────────────────────────────────┘

OPCIÓN 1 - Usar archivo .env (RECOMENDADO):
    1. Cree un archivo .env en el mismo directorio
    2. Agregue: OPENAI_API_KEY=sk-...
    3. Instale: pip install python-dotenv

OPCIÓN 2 - Configurar manualmente (SOLO PARA PRUEBAS):
    import os
    os.environ['OPENAI_API_KEY'] = 'sk-...'
    # o
    os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'

💡 Beneficios de LangGraph:
  ✓ Análisis contextual mucho más sofisticado
  ✓ Justificaciones egiptológicas detalladas con base de 80+ determinativos
  ✓ Identificación de casos ambiguos
  ✓ Razonamiento sobre significado semántico
  ✓ Control de flujo explícito entre agentes

⚠️  Si no configura una API key, el código funcionará normalmente usando
   solo análisis basado en reglas (¡todavía muy eficaz!).

""")

# ============================================================================
# CONFIGURACIÓN SEGURA DE API KEYS
# ============================================================================

import os

# Intentar cargar variables de archivo .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Variables cargadas desde archivo .env")
except ImportError:
    print("💡 Dica: instale python-dotenv para usar archivo .env")
    print("   pip install python-dotenv")
except Exception:
    pass

print("✅ Configuração de segurança concluída!\n")

# ============================================================================
# ⚙️ CONFIGURAÇÕES DO USUÁRIO
# ============================================================================

# 🎯 CONFIGURAÇÃO: QUANTIDADE E RANGE DE FRASES A PROCESSAR

# ═══════════════════════════════════════════════════════════════
# MODO 1: PROCESSAR QUANTIDADE ESPECÍFICA (do início)
# ═══════════════════════════════════════════════════════════════
NUM_FRASES_PROCESSAR = 100  # Número específico ou None (todas)

# ═══════════════════════════════════════════════════════════════
# MODO 2: PROCESSAR RANGE ESPECÍFICO (de X até Y)
# ═══════════════════════════════════════════════════════════════
# Ative este modo definindo INICIO_RANGE e FIM_RANGE
# Deixe como None para desativar

INICIO_RANGE = None  # ← Frase inicial (ex: 50)
FIM_RANGE = None     # ← Frase final (ex: 100)

# Exemplos:
#   Processar frases 50 a 100:
#     INICIO_RANGE = 50
#     FIM_RANGE = 100
#
#   Processar frases 0 a 50:
#     INICIO_RANGE = 0
#     FIM_RANGE = 50
#
#   Processar últimas 50 frases:
#     INICIO_RANGE = -50
#     FIM_RANGE = None

# ═══════════════════════════════════════════════════════════════
# NOTAS:
# - Se INICIO_RANGE e FIM_RANGE estiverem definidos, eles têm PRIORIDADE
# - Caso contrário, usa NUM_FRASES_PROCESSAR
# - Índices começam em 0 (primeira frase = 0)
# ═══════════════════════════════════════════════════════════════

# Mensagem de configuração
if INICIO_RANGE is not None or FIM_RANGE is not None:
    inicio_msg = INICIO_RANGE if INICIO_RANGE is not None else 0
    fim_msg = FIM_RANGE if FIM_RANGE is not None else "fim"
    print(f"⚙️  CONFIGURAÇÃO: Processar RANGE de frases {inicio_msg} a {fim_msg}")
elif NUM_FRASES_PROCESSAR is not None:
    print(f"⚙️  CONFIGURAÇÃO: Processar {NUM_FRASES_PROCESSAR} frases (do início)")
else:
    print("⚙️  CONFIGURAÇÃO: Processar TODAS as frases")
print()


#%%
# ============================================================================
# FASE 1 - IMPORTS E CONFIGURAÇÃO
# ============================================================================

import random, json, re
import numpy as np
import pandas as pd
#import torch
import matplotlib.pyplot as plt

#from ast import literal_eval
from collections import defaultdict, Counter
from typing import TypedDict, List, Dict, Any

# ── Top-level conditional imports (avoid linter "undefined name" warnings) ──
try:
    import optuna  # type: ignore
    _HAS_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore
    _HAS_OPTUNA = False
#import operator
#from typing_extensions import TypedDict

# ── v1.42: Centralised seed — change ONE constant to replay any experiment ───
GLOBAL_SEED: int = 42
# Covers: random, numpy, outer CV splits, inner CV splits, all Optuna samplers.
# Anthropic LLM calls use temperature=0.0 (no seed= API param available).
# OpenAI calls use temperature=0.0 + seed=GLOBAL_SEED (fully deterministic).
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
#torch.manual_seed(GLOBAL_SEED)
#if torch.cuda.is_available():
#    torch.cuda.manual_seed_all(GLOBAL_SEED)

# Configurações globais
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def select_expert_examples(df, n_examples=5):
    """
    Seleciona exemplos diversos do especialista para Few-Shot Learning
    
    Args:
        df: DataFrame com análises do especialista
        n_examples: número de exemplos a selecionar
        
    Returns:
        Lista de tuplas (code, translation)
    """
    # Filtrar exemplos de qualidade
    quality = df[
        (df['translation'].notna()) & 
        (df['translation'].str.len() > 40)
    ].copy()
    
    if len(quality) < n_examples:
        print(f"⚠️  Apenas {len(quality)} exemplos disponíveis")
        n_examples = len(quality)
    
    # Selecionar exemplos de diferentes tamanhos (diversidade)
    quality['code_len'] = quality['Code'].str.count('-') + 1
    quality = quality.sort_values('code_len')
    
    # Distribuir uniformemente
    step = max(1, len(quality) // n_examples)
    selected = []
    
    for i in range(n_examples):
        idx = min(i * step, len(quality) - 1)
        row = quality.iloc[idx]
        selected.append((row['Code'], row['translation']))
    
    print(f"✅ Few-Shot: {len(selected)} exemplos selecionados")
    return selected

def format_expert_examples(examples):
    """
    Formata exemplos do especialista para inclusão no prompt
    
    Args:
        examples: lista de tuplas (code, translation)
        
    Returns:
        String formatada para o prompt
    """
    if not examples:
        return ""
    
    text = "EJEMPLOS DE ANÁLISIS DEL ESPECIALISTA:\n\n"
    
    for i, (code, translation) in enumerate(examples, 1):
        # Truncar se muito longo
        code_display = code[:70] + "..." if len(code) > 70 else code
        trans_display = translation[:120] + "..." if len(translation) > 120 else translation
        
        text += f"Ejemplo {i}:\n"
        text += f"  MdC: {code_display}\n"
        text += f"  Interpretación: {trans_display}\n\n"
    
    text += "IMPORTANTE: Estos ejemplos muestran el tipo de análisis esperado. "
    text += "Use este contexto para su análisis.\n\n"
    text += "="*70 + "\n\n"
    
    return text

# Variável global para armazenar exemplos (será inicializada após carregar dados)


EXPERT_EXAMPLES = []


# ============================================================================
# V1.35 — AGENT-SPECIFIC GUIDANCE FUNCTIONS
# ============================================================================

def format_params_for_semanticist(params: Dict[str, float]) -> str:
    """
    V1.35 — Guidance for the SEMANTICISTA agent.
    Covers: context-dependent signs, high-risk signs, phonogram defaults,
    and disambiguation rules. All guidance relates to MEANING, not position.
    The Semanticista cannot see structural position, so positional bonuses
    are intentionally excluded.
    """
    if not params:
        return ""
    lines = [
        "=" * 65,
        "SEMANTIC CALIBRATION GUIDANCE",
        "(data-validated; apply to meaning-based decisions only)",
        "=" * 65,
    ]

    # Context-dependent signs — purely semantic decision
    ctx_signs = {
        'W24': ('vessel/cup',       'food, offering, liquid, drink'),
        'R8':  ('standard on pole', 'divine, ritual, sacred, temple'),
        'D4':  ('eye',              'seeing, vision, watching, perceiving'),
        'F4':  ('forepart of lion', 'front, beginning, leader'),
        'I9':  ('horned viper',     'rarely determinative — phonogram f is primary role'),
        'L1':  ('dung beetle',      'rebirth, becoming — only in Kheper contexts'),
        'K1':  ('fish',             'fish, water creature — aquatic context required'),
        'X1':  ('bread loaf',       'food/offering context required for determinative use'),
    }
    thresh_lines = []
    for sign, (meaning, ctx) in ctx_signs.items():
        key = f'thresh_{sign}'
        if key in params:
            t = params[key]
            thresh_lines.append(
                f"  {sign} ({meaning}): min {t*100:.0f}% confidence "
                f"AND context must include [{ctx}]")
    if thresh_lines:
        lines.append("\nCONTEXT-DEPENDENT SIGNS (require semantic match + threshold):")
        lines.extend(thresh_lines)
        lines.append("  -> Without matching semantic context: REJECT regardless of position.")

    # High-risk signs — semantic evidence required
    prc = params.get('penalty_recently_critical', 1.0)
    prm = params.get('penalty_recently_minor',    1.0)
    if prc < 0.85:
        lines.append(f"\nHIGH-RISK SIGNS (score x{prc:.2f} — strong semantic evidence required):")
        lines.append("  I9 (horned viper/f), I10 (cobra/dj), L1 (dung beetle/Kheper)")
        lines.append("  -> DEFAULT = REJECT. Only accept with explicit, unambiguous semantic")
        lines.append("     support specific to the sign's determinative meaning.")
    if prm < 0.90:
        lines.append(f"\nMODERATE-RISK SIGNS (score x{prm:.2f} — clear semantic support needed):")
        lines.append("  D58 (foot/b), U1 (sickle), L2 (bee), I5 (lotus), I6")
        lines.append("  -> Require clear context match. Default to REJECT if uncertain.")

    # Phonogram-primary signs — semantic meaning is phonetic
    ppp = params.get('penalty_phonogram_primary_mult', 1.0)
    if ppp < 0.70:
        lines.append(f"\nPHONOGRAM-PRIMARY SIGNS (score x{ppp:.2f} — semantic function = phonetic):")
        lines.append("  M17 (reed/i), D21 (mouth/r), G43 (quail/w), V31 (basket/k),")
        lines.append("  N35 (water/n), S29 (folded cloth/s), G17 (owl/m), X1 (bread/t)")
        lines.append("  -> These signs convey SOUND, not meaning, in most contexts.")
        lines.append("     REJECT as determinative unless translation clearly requires it.")

    # Phonetic complement
    ppc = params.get('penalty_phonetic_complement', 0.0)
    if ppc > 0.05:
        lines.append(f"\nPHONETIC COMPLEMENTS — NOT determinatives:")
        lines.append("  If a sign repeats the phonetic value of the preceding root,")
        lines.append("  it is a phonetic complement. REJECT as determinative.")

    lines.append("=" * 65)
    return "\n".join(lines)


def format_params_for_structuralist(params: Dict[str, float]) -> str:
    """
    V1.35 — Guidance for the ESTRUTURALISTA agent.
    Covers: positional bonuses, reliable signs, plural markers, and
    structural patterns. All guidance relates to POSITION and STRUCTURE,
    not meaning. The Estruturalista cannot see translation context, so
    semantic thresholds are intentionally excluded.
    """
    if not params:
        return ""
    lines = [
        "=" * 65,
        "STRUCTURAL CALIBRATION GUIDANCE",
        "(data-validated; apply to position-based decisions only)",
        "=" * 65,
    ]

    # Positional preferences
    bf  = params.get('bonus_final',        0.0)
    bgf = params.get('bonus_global_final', 0.0)
    bne = params.get('bonus_near_end',     0.0)
    bge = params.get('bonus_group_end',    0.0)
    lines.append("\nPOSITIONAL PREFERENCE (data-validated from this corpus):")
    if bf > 0.05:
        lines.append(f"  - END of a word group (before '-' separator): STRONGLY PREFERRED "
                     f"(+{bf*100:.0f}pt). This is the most reliable structural indicator.")
    if bgf > 0.03:
        lines.append(f"  - VERY LAST sign of the entire sequence: MOST PREFERRED "
                     f"(+{bgf*100:.0f}pt). Accept if in KB and not an obvious phonogram.")
    if bne > 0.03:
        lines.append(f"  - LAST 3 POSITIONS of the full sequence: PREFERRED "
                     f"(+{bne*100:.0f}pt). Extra weight for these candidates.")
    if bge > 0.02:
        lines.append(f"  - End of a GROUP (just before ':'): slightly preferred "
                     f"(+{bge*100:.0f}pt).")
    lines.append("  -> When structural evidence is ambiguous, prefer the more final sign.")

    # Reliable signs
    br = params.get('bonus_reliable', 0.0)
    if br > 0.05:
        lines.append(f"\nSTRUCTURALLY RELIABLE SIGNS (+{br*100:.0f}pt validated bonus):")
        lines.append("  A1, B1, A40, Z1, Z2, Z3, Q5, O49, A24, N1, A193")
        lines.append("  -> When these signs appear at or near word-final position,")
        lines.append("     DEFAULT = ACCEPT as determinative.")

    # Plural markers
    bpm = params.get('bonus_plural_marker', 0.0)
    bph = params.get('bonus_plural_head',   0.0)
    if bpm > 0.02:
        lines.append(f"\nPLURAL MARKERS — accept when structurally final in noun group "
                     f"(+{bpm*100:.0f}pt):")
        lines.append("  Z1 (|), Z2 (||), Z3 (|||) — the three stroke-count signs")
        lines.append("  -> At the end of a noun word-group = almost always determinative.")
    if bph > 0.02:
        lines.append(f"  Plural HEAD signs (+{bph*100:.0f}pt when heading a plural group):")
        lines.append("  -> Sign immediately before Z1/Z2/Z3 cluster also gets bonus.")

    # Strong determinatives
    bsd = params.get('bonus_strong_det', 0.0)
    if bsd > 0.02:
        lines.append(f"\nSTRONG STRUCTURAL DETERMINATIVES (+{bsd*100:.0f}pt):")
        lines.append("  A1 (seated man), B1 (seated woman), A40 (god), D54 (walking legs)")
        lines.append("  N1 (sky), O49 (town), Q5 (coffin)")
        lines.append("  -> These signs have strong structural determinative function.")
        lines.append("     Accept when in plausible position even with moderate confidence.")

    # Phonetic complement warning (structural)
    ppc = params.get('penalty_phonetic_complement', 0.0)
    if ppc > 0.05:
        lines.append(f"\nPHONETIC COMPLEMENTS — structural note:")
        lines.append("  Signs that immediately follow a root and repeat its final phoneme")
        lines.append("  are phonetic complements, NOT determinatives.")
        lines.append("  Structurally: phonetic complements appear BEFORE the final position.")
        lines.append("  -> If a sign appears mid-word with no structural break before it,")
        lines.append("     treat it as a complement, not a determinative.")

    lines.append("=" * 65)
    return "\n".join(lines)


def format_params_for_synthesizer(params: Dict[str, float]) -> str:
    """
    V1.35 — Guidance for the SINTETIZADOR agent.
    The Sintetizador sees BOTH semantic and structural analyses and makes
    the final decision. It receives the full picture plus explicit
    tie-breaking logic derived from the optimised parameter weights.
    """
    if not params:
        return ""

    w_sem    = params.get('w_sem_llm',    0.1)
    w_struct = params.get('w_struct_llm', 0.1)
    w_rule   = params.get('w_rule_llm',   0.8)
    b2       = params.get('bonus_2src',   0.08)
    b3       = params.get('bonus_3src',   0.06)
    prc      = params.get('penalty_recently_critical', 1.0)
    br       = params.get('bonus_reliable', 0.0)

    lines = [
        "=" * 65,
        "SYNTHESIS CALIBRATION GUIDANCE",
        "(data-validated; apply when reconciling semantic + structural)",
        "=" * 65,
    ]

    # Source weighting for arbitration
    lines.append("\nSOURCE WEIGHTING (use to arbitrate disagreements):")
    lines.append(f"  Semantic agent weight:    {w_sem:.2f}")
    lines.append(f"  Structural agent weight:  {w_struct:.2f}")
    lines.append(f"  Rule-based weight:        {w_rule:.2f}")
    total = w_sem + w_struct + w_rule
    dominant = max([('semantic', w_sem), ('structural', w_struct), ('rules', w_rule)],
                   key=lambda x: x[1])
    lines.append(f"  -> Dominant source: {dominant[0]} ({dominant[1]/total*100:.0f}% of total weight)")
    lines.append("     When agents disagree, lean toward the dominant source.")

    # Multi-source agreement bonus
    if b2 > 0.0 or b3 > 0.0:
        lines.append(f"\nAGREEMENT BONUS:")
        lines.append(f"  2 sources agree: +{b2*100:.0f}pt — significant signal, lean toward ACCEPT")
        lines.append(f"  3 sources agree: +{b3*100:.0f}pt — strong signal, ACCEPT unless clear error")
        lines.append("  -> If semantic + structural + rules all agree: ACCEPT without hesitation.")
        lines.append("  -> If only 1 source: apply strict threshold before accepting.")

    # Reliable signs — final arbitration
    if br > 0.05:
        lines.append(f"\nRELIABLE SIGNS — accept when any source supports (+{br*100:.0f}pt):")
        lines.append("  A1, B1, A40, Z1, Z2, Z3, Q5, O49, A24, N1")
        lines.append("  -> If the rule system found these AND position is plausible: ACCEPT.")
        lines.append("     Do not reject reliable signs on weak semantic/structural doubts.")

    # High-risk final veto
    if prc < 0.85:
        lines.append(f"\nHIGH-RISK FINAL VETO (score x{prc:.2f}):")
        lines.append("  I9, I10, L1 — even if both agents suggest these, apply extra scrutiny.")
        lines.append("  -> Accept ONLY if: both agents agree AND rule system supports AND")
        lines.append("     confidence > 0.85 from both. Otherwise REJECT.")

    # Tie-breaking rules
    lines.append("\nTIE-BREAKING RULES (when agents conflict):")
    lines.append("  1. Grammar/plural signs (Z1, Z2, Z3): STRUCTURAL wins over semantic.")
    lines.append("     Position at end of noun group is sufficient evidence.")
    lines.append("  2. Context-dependent signs (W24, R8, D4): SEMANTIC wins over structural.")
    lines.append("     No amount of positional evidence compensates for wrong semantic context.")
    lines.append("  3. High-frequency person signs (A1, A40, B1): accept when EITHER agrees.")
    lines.append("  4. Unknown/variant signs (not in KB): require BOTH agents to agree.")
    lines.append("     Single-source prediction for unknown signs -> REJECT.")

    # V1.37 Fix 6: Sintetizador reconsiders empty when rule baseline has candidates
    lines.append("\nV1.37 EMPTY-OUTPUT SAFEGUARD:")
    lines.append("  Before outputting EMPTY [], check: does the rule-based baseline")
    lines.append("  predict at least one determinative for this sequence?")
    lines.append("  If YES and BOTH agents returned empty, this is a warning sign.")
    lines.append("  In this case: reconsider the sequence. Look at the baseline candidates.")
    lines.append("  If the baseline candidate is a RELIABLE sign (A1, Z1, Z2, A40, B1, A24)")
    lines.append("  and at least one agent mentioned it as an alternative — ACCEPT it.")
    lines.append("  Do NOT blindly follow both agents to empty when rules found candidates.")
    lines.append("  Exception: if both agents explicitly argued AGAINST all baseline candidates,")
    lines.append("  then empty output is justified.")

    lines.append("\nV1.37 INDEPENDENT EVALUATION REMINDER:")
    lines.append("  If you rejected I9, L1, I10 or other phonograms — good. Now continue")
    lines.append("  evaluating the OTHER signs in the sequence normally. Phonogram rejection")
    lines.append("  is LOCAL and must NOT suppress evaluation of A1, Z1, A53, Y2, etc.")

    lines.append("=" * 65)
    return "\n".join(lines)


# ============================================================================
# V1.29 - FUNÇÕES DE GROUND TRUTH E AVALIAÇÃO LOO
# ============================================================================

def parse_ground_truth_from_docx(docx_path: str) -> list:
    """
    Extrai o gabarito de determinativos de um arquivo DOCX anotado pelo especialista.

    Regra de extração:
      • Parágrafos que começam com "Sequência:" contêm a sequência MdC.
      • Os signos em NEGRITO (bold) nesses parágrafos são os determinativos corretos.
      • Separadores ('-', ':', '*', ' ') em negrito são ignorados.

    Returns:
        Lista de dicts: [{'raw_seq': str, 'gt_determinatives': [str, ...]}, ...]
    """
    try:
        from docx import Document
    except ImportError:
        print("⚠️  python-docx não instalado. Execute: pip install python-docx")
        return []

    results = []
    try:
        doc = Document(docx_path)
    except Exception as e:
        print(f"⚠️  Erro ao abrir {docx_path}: {e}")
        return []

    SEPARATOR_TOKENS = {'-', ':', '*', ' ', '–', '—'}

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text.startswith('Sequência:'):
            continue
        raw_seq = text.replace('Sequência:', '').strip()
        bold_signs = []
        for run in para.runs:
            if run.bold:
                tok = run.text.strip()
                if not tok or tok in SEPARATOR_TOKENS:
                    continue
                # A bold run might span multiple signs joined by '-' (e.g., "Z1-Z1-Z1-Z1")
                # Split on '-' to get individual sign codes
                sub_signs = [s.strip() for s in tok.split('-') if s.strip() and s.strip() not in SEPARATOR_TOKENS]
                bold_signs.extend(sub_signs)
        results.append({'raw_seq': raw_seq, 'gt_determinatives': bold_signs})

    return results


def load_loo_ground_truth(docx_paths: list) -> dict:
    """
    Carrega o gabarito LOO de múltiplos arquivos DOCX e retorna um dicionário
    mapeando a sequência MdC (normalizada) → lista de determinativos corretos.

    A chave é a sequência com os ** removidos e espaços eliminados.
    O valor é a lista de signos em negrito (com repetições preservadas).

    Returns:
        dict: {normalized_seq: [sign, sign, ...]}
    """
    all_entries = []
    for path in docx_paths:
        entries = parse_ground_truth_from_docx(path)
        all_entries.extend(entries)
        print(f"✅ GT carregado de {path}: {len(entries)} sequências")

    gt_dict = {}
    for entry in all_entries:
        raw = entry['raw_seq']
        # Inline normalization (normalize_mdc may not be defined yet at call time)
        norm = re.sub(r'\s+', '', raw.strip())
        # Se já existe, usa a versão com mais anotações (mais completa)
        if norm not in gt_dict or len(entry['gt_determinatives']) > len(gt_dict[norm]):
            gt_dict[norm] = entry['gt_determinatives']

    print(f"✅ Total GT: {len(gt_dict)} sequências únicas")
    return gt_dict


def _jaccard_sign_overlap(seq_a: str, seq_b: str) -> float:
    """
    Computes Jaccard similarity between the sign vocabularies of two MdC sequences.
    Separators and empty strings are excluded.
    Used by select_expert_examples_loo (v1.33) for similarity-based selection.
    """
    SEPS = {'-', ':', '*', ''}
    def signs(s):
        return {t for t in s.split('-') if t not in SEPS}
    a, b = signs(str(seq_a)), signs(str(seq_b))
    if not a | b:
        return 0.0
    return len(a & b) / len(a | b)


def select_expert_examples_loo(df_other: 'pd.DataFrame',
                                n_examples: int = 5,
                                test_sequence: str = None) -> list:
    """
    V1.33 — Similarity-based few-shot selection.

    If test_sequence is provided, selects the n_examples training sequences
    with the highest Jaccard sign-vocabulary overlap with the test sequence.
    This gives the LLM the most relevant analogies — sequences that share
    the same sign inventory are more informative examples than random ones.

    Fallback (test_sequence=None): original length-diversity strategy.

    Why Jaccard over embedding similarity?
      • No extra API call needed — purely symbolic comparison
      • Signs are the units of meaning here, not words
      • Fast and deterministic
    """
    quality = df_other[
        (df_other['translation'].notna()) &
        (df_other['translation'].str.len() > 40)
    ].copy()

    if len(quality) == 0:
        return []
    if len(quality) < n_examples:
        n_examples = len(quality)

    if test_sequence is not None:
        # V1.33: rank by Jaccard similarity to the test sequence
        quality = quality.copy()
        quality['_sim'] = quality['Code'].apply(
            lambda c: _jaccard_sign_overlap(str(c), test_sequence)
            if isinstance(c, str) else 0.0
        )
        # Take top-n by similarity; break ties by translation length (more informative)
        quality = quality.sort_values(['_sim', 'translation'],
                                      ascending=[False, False])
        selected = []
        for _, row in quality.head(n_examples).iterrows():
            selected.append((row['Code'], row['translation']))
        return selected
    else:
        # Original strategy: diversity across sequence lengths
        quality['code_len'] = quality['Code'].str.count('-') + 1
        quality = quality.sort_values('code_len')
        step = max(1, len(quality) // n_examples)
        selected = []
        for i in range(n_examples):
            idx = min(i * step, len(quality) - 1)
            row = quality.iloc[idx]
            selected.append((row['Code'], row['translation']))
        return selected


def _signs_to_multiset(sign_list: list) -> dict:
    """Converte lista de signos para multiconjunto (dicionário sign → count)."""
    from collections import Counter
    return dict(Counter(sign_list))


def evaluate_single_prediction(predicted_signs: list, gt_signs: list) -> dict:
    """
    Avalia uma predição individual por comparação de multiconjuntos.

    V1.31 — CORREÇÃO DO CASO NULL:
    Quando pred=[] e GT=[], o cálculo TP/FP/FN clássico produz 0/0 → F1=0%,
    penalizando uma predição CORRECTA. Este função distingue quatro casos:

    ┌─────────────┬───────────┬────────────────────────────────────────────┐
    │ Tipo        │ F1        │ Significado                                │
    ├─────────────┼───────────┼────────────────────────────────────────────┤
    │ TRUE_NULL   │ 1.0  ✅   │ pred=∅ & GT=∅  — ausência correctamente   │
    │             │           │ prevista; não entra no denominador micro   │
    ├─────────────┼───────────┼────────────────────────────────────────────┤
    │ FP_NULL     │ 0.0  ❌   │ pred≠∅ & GT=∅  — sistema inventou signos  │
    ├─────────────┼───────────┼────────────────────────────────────────────┤
    │ FN_NULL     │ 0.0  ❌   │ pred=∅ & GT≠∅  — sistema perdeu tudo      │
    ├─────────────┼───────────┼────────────────────────────────────────────┤
    │ STANDARD    │ normal    │ pred≠∅ & GT≠∅  — cálculo TP/FP/FN normal  │
    └─────────────┴───────────┴────────────────────────────────────────────┘
    """
    pred_empty = (len(predicted_signs) == 0)
    gt_empty   = (len(gt_signs) == 0)

    if pred_empty and gt_empty:
        return {'tp': 0, 'fp': 0, 'fn': 0,
                'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                'outcome_type': 'TRUE_NULL',
                'null_correct': True, 'fp_null': False, 'fn_null': False}

    if not pred_empty and gt_empty:
        return {'tp': 0, 'fp': len(predicted_signs), 'fn': 0,
                'precision': 0.0, 'recall': 1.0, 'f1': 0.0,
                'outcome_type': 'FP_NULL',
                'null_correct': False, 'fp_null': True, 'fn_null': False}

    if pred_empty and not gt_empty:
        return {'tp': 0, 'fp': 0, 'fn': len(gt_signs),
                'precision': 1.0, 'recall': 0.0, 'f1': 0.0,
                'outcome_type': 'FN_NULL',
                'null_correct': False, 'fp_null': False, 'fn_null': True}

    # STANDARD: ambos não vazios
    pred_ms   = _signs_to_multiset(predicted_signs)
    gt_ms     = _signs_to_multiset(gt_signs)
    all_signs = set(pred_ms) | set(gt_ms)
    tp = sum(min(pred_ms.get(s, 0), gt_ms.get(s, 0)) for s in all_signs)
    fp = sum(max(0, pred_ms.get(s, 0) - gt_ms.get(s, 0)) for s in all_signs)
    fn = sum(max(0, gt_ms.get(s, 0) - pred_ms.get(s, 0)) for s in all_signs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'outcome_type': 'STANDARD',
            'null_correct': False, 'fp_null': False, 'fn_null': False}


def compute_loo_metrics(loo_results: list) -> dict:
    """
    V1.31 — NULL-AWARE metric aggregation.

    MICRO-AVERAGE: sums TP/FP/FN over all samples.
      True Nulls have TP=FP=FN=0, so they don't affect the denominator.
      Micro-F1 therefore reflects performance specifically on sequences
      that have determinatives.

    MACRO-AVERAGE corrected (v1.31 default — use this for reporting):
      Mean per-sequence F1 where True Nulls contribute F1=1.0.
      Correctly predicting absence is a correct prediction and should score 1.

    MACRO-AVERAGE classic (for comparison with published literature):
      True Nulls excluded entirely from the mean.

    NULL STATISTICS:
      Null Accuracy  = fraction of GT=∅ sequences correctly predicted empty.
      FN-Null Rate   = fraction of GT≠∅ sequences where ALL signs were missed.
    """
    import numpy as np
    from collections import defaultdict

    n = len(loo_results)
    true_nulls = [r for r in loo_results if r['metrics']['outcome_type'] == 'TRUE_NULL']
    fp_nulls   = [r for r in loo_results if r['metrics']['outcome_type'] == 'FP_NULL']
    fn_nulls   = [r for r in loo_results if r['metrics']['outcome_type'] == 'FN_NULL']
    standards  = [r for r in loo_results if r['metrics']['outcome_type'] == 'STANDARD']
    gt_empty_total    = len(true_nulls) + len(fp_nulls)
    gt_nonempty_total = len(fn_nulls)   + len(standards)

    # Micro
    total_tp = sum(r['metrics']['tp'] for r in loo_results)
    total_fp = sum(r['metrics']['fp'] for r in loo_results)
    total_fn = sum(r['metrics']['fn'] for r in loo_results)
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2*micro_p*micro_r / (micro_p+micro_r) if (micro_p+micro_r) > 0 else 0.0

    # Macro corrected (True Nulls = F1 1.0)
    macro_f1 = float(np.mean([r['metrics']['f1']        for r in loo_results])) if n else 0.0
    macro_p  = float(np.mean([r['metrics']['precision'] for r in loo_results])) if n else 0.0
    macro_r  = float(np.mean([r['metrics']['recall']    for r in loo_results])) if n else 0.0

    # Macro classic (excludes True Nulls)
    non_null = [r for r in loo_results if r['metrics']['outcome_type'] != 'TRUE_NULL']
    if non_null:
        macro_f1_c = float(np.mean([r['metrics']['f1']        for r in non_null]))
        macro_p_c  = float(np.mean([r['metrics']['precision'] for r in non_null]))
        macro_r_c  = float(np.mean([r['metrics']['recall']    for r in non_null]))
    else:
        macro_f1_c = macro_p_c = macro_r_c = 0.0

    # Per-sign
    sign_tp = defaultdict(int); sign_fp = defaultdict(int); sign_fn = defaultdict(int)
    for r in loo_results:
        pm = _signs_to_multiset(r['predicted_signs'])
        gm = _signs_to_multiset(r['gt_signs'])
        for s in set(pm) | set(gm):
            sign_tp[s] += min(pm.get(s, 0), gm.get(s, 0))
            sign_fp[s] += max(0, pm.get(s, 0) - gm.get(s, 0))
            sign_fn[s] += max(0, gm.get(s, 0) - pm.get(s, 0))
    sign_metrics = {}
    for s in sorted(set(sign_tp) | set(sign_fp) | set(sign_fn)):
        tp_s, fp_s, fn_s = sign_tp[s], sign_fp[s], sign_fn[s]
        p_s = tp_s/(tp_s+fp_s) if (tp_s+fp_s) > 0 else 0.0
        r_s = tp_s/(tp_s+fn_s) if (tp_s+fn_s) > 0 else 0.0
        f_s = 2*p_s*r_s/(p_s+r_s) if (p_s+r_s) > 0 else 0.0
        sign_metrics[s] = {'tp':tp_s,'fp':fp_s,'fn':fn_s,
                           'precision':p_s,'recall':r_s,'f1':f_s}

    return {
        'micro': {'precision': micro_p, 'recall': micro_r, 'f1': micro_f1,
                  'total_tp': total_tp, 'total_fp': total_fp, 'total_fn': total_fn,
                  'note': 'Sobre signos; True Nulls não alteram TP/FP/FN'},
        'macro': {'precision': macro_p, 'recall': macro_r, 'f1': macro_f1,
                  'note': 'Média por sequência; True Nulls incluídos como F1=1.0 ✅ (v1.31)'},
        'macro_classic': {'precision': macro_p_c, 'recall': macro_r_c, 'f1': macro_f1_c,
                          'note': 'Exclui True Nulls (convenção clássica; para comparação)'},
        'null_stats': {
            'n_true_null':       len(true_nulls),
            'n_fp_null':         len(fp_nulls),
            'n_fn_null':         len(fn_nulls),
            'n_standard':        len(standards),
            'gt_empty_total':    gt_empty_total,
            'gt_nonempty_total': gt_nonempty_total,
            'null_accuracy':     len(true_nulls)/gt_empty_total    if gt_empty_total    > 0 else None,
            'fn_null_rate':      len(fn_nulls)/gt_nonempty_total   if gt_nonempty_total > 0 else None,
        },
        'per_sign': sign_metrics,
        'n_samples': n,
    }


def print_loo_report(metrics: dict, loo_results: list) -> None:
    """Imprime relatório formatado dos resultados LOO (v1.31 null-aware)."""
    print("\n" + "═"*70)
    print("📊  RESULTADOS LEAVE-ONE-OUT (LOO) — v1.31 (null-aware + optimization)")
    print("═"*70)

    ns = metrics.get('null_stats', {})
    print(f"\nDISTRIBUIÇÃO DE RESULTADOS ({metrics['n_samples']} amostras):")
    print(f"  ✅ TRUE_NULL  (pred=∅, GT=∅): {ns.get('n_true_null','?'):>4}  — ausência correctamente prevista")
    print(f"  ❌ FP_NULL   (pred≠∅, GT=∅): {ns.get('n_fp_null','?'):>4}  — sistema inventou determinativos")
    print(f"  ❌ FN_NULL   (pred=∅, GT≠∅): {ns.get('n_fn_null','?'):>4}  — sistema perdeu todos os determinativos")
    print(f"  🔵 STANDARD  (pred≠∅, GT≠∅): {ns.get('n_standard','?'):>4}  — avaliação TP/FP/FN normal")

    na = ns.get('null_accuracy'); fnr = ns.get('fn_null_rate')
    print(f"\n  Null Accuracy  (% de GT=∅ correctamente previstas): "
          + (f"{na*100:.1f}%" if na is not None else "N/A"))
    print(f"  FN-Null Rate   (% de GT≠∅ totalmente perdidas)    : "
          + (f"{fnr*100:.1f}%" if fnr is not None else "N/A"))

    m = metrics['micro']; ma = metrics['macro']; mc = metrics.get('macro_classic', {})

    print(f"\nMICRO-AVERAGE (sobre signos individuais):")
    print(f"  Precision : {m['precision']*100:.2f}%")
    print(f"  Recall    : {m['recall']*100:.2f}%")
    print(f"  F1-Score  : {m['f1']*100:.2f}%")
    print(f"  TP={m['total_tp']}  FP={m['total_fp']}  FN={m['total_fn']}")

    print(f"\nMACRO-AVERAGE CORRIGIDO ✅ (True Nulls = F1 1.0  — métrica principal):")
    print(f"  Precision : {ma['precision']*100:.2f}%")
    print(f"  Recall    : {ma['recall']*100:.2f}%")
    print(f"  F1-Score  : {ma['f1']*100:.2f}%")

    print(f"\nMACRO-AVERAGE CLÁSSICO (exclui True Nulls — para comparação com literatura):")
    print(f"  Precision : {mc.get('precision',0)*100:.2f}%")
    print(f"  Recall    : {mc.get('recall',0)*100:.2f}%")
    print(f"  F1-Score  : {mc.get('f1',0)*100:.2f}%")

    ps = metrics['per_sign']
    sorted_signs = sorted(ps.items(), key=lambda x: -(x[1]['tp'] + x[1]['fn']))
    print(f"\nTOP-20 SIGNOS (por TP+FN):")
    print(f"  {'Sign':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}")
    print("  " + "-"*52)
    for sign, sm in sorted_signs[:20]:
        print(f"  {sign:<12} {sm['tp']:>4} {sm['fp']:>4} {sm['fn']:>4} "
              f"{sm['precision']*100:>6.1f}% {sm['recall']*100:>6.1f}% {sm['f1']*100:>6.1f}%")

    # Worst only from STANDARD (True/FP/FN-Nulls are structurally different cases)
    worst = sorted([r for r in loo_results if r['metrics']['outcome_type'] == 'STANDARD'],
                   key=lambda r: r['metrics']['f1'])
    print(f"\n5 PIORES AMOSTRAS STANDARD (pred≠∅ & GT≠∅):")
    for r in worst[:5]:
        mm = r['metrics']
        print(f"  SEQ: {r['sequence'][:55]}")
        print(f"    Pred={r['predicted_signs']}  GT={r['gt_signs']}")
        print(f"    P={mm['precision']*100:.1f}%  R={mm['recall']*100:.1f}%  F1={mm['f1']*100:.1f}%")
    print("═"*70 + "\n")


# ============================================================================
# V1.30 - OTIMIZAÇÃO DE HIPERPARÂMETROS VIA LOO + OPTUNA
# ============================================================================
#
# ESTRATÉGIA:
#   Stage 1 (rápido, sem LLM):
#     • Pré-computa evidências de regras para todas as 99 amostras uma vez.
#     • Optuna busca os melhores valores para ~25 parâmetros usando LOO-F1
#       como função objetivo (cada avaliação leva ~milissegundos).
#     • Os melhores parâmetros são salvos em JSON e aplicados globalmente.
#   Stage 2 (LOO_MODE=True, com LLM):
#     • Usa os parâmetros otimizados do Stage 1 no pipeline completo.
# ============================================================================

# ── Espaço de parâmetros (bounds para Optuna) ────────────────────────────────
PARAM_SPACE: Dict[str, tuple] = {
    # ── Rule-path source weight (only source active during inner CV) ──────────
    # In rule-only mode conf_sem=conf_struct=0, so only w_rule matters.
    # Kept as single value; w_sem and w_struct are legacy names for reference.
    'w_sem':    (0.10, 1.00),   # used in LLM path via w_sem_llm below
    'w_struct': (0.10, 1.00),   # used in LLM path via w_struct_llm below
    'w_rule':   (0.10, 1.00),   # primary weight in rule-only inner CV

    # ── Option D: LLM-specific source weights (separate from rule-path) ───────
    # Not optimised in inner CV (no LLM data there); initialised from flat prior.
    # Used exclusively during test phase when USE_LANGGRAPH=True.
    'w_sem_llm':    (0.05, 0.70),
    'w_struct_llm': (0.05, 0.70),
    'w_rule_llm':   (0.05, 0.70),

    # Bônus multi-fonte
    'bonus_2src': (0.00, 0.20),
    'bonus_3src': (0.00, 0.15),

    # Bônus posicionais
    'bonus_final':        (0.00, 0.30),
    'bonus_global_final': (0.00, 0.20),
    'bonus_near_end':     (0.00, 0.20),
    'bonus_group_end':    (0.00, 0.15),

    # Bônus por categoria
    'bonus_reliable':      (0.00, 0.30),
    'bonus_plural_marker': (0.00, 0.20),
    'bonus_plural_head':   (0.00, 0.20),
    'bonus_strong_det':    (0.00, 0.20),

    # Penalidades em _score_candidate
    'penalty_phonogram_primary_mult': (0.10, 1.00),
    'penalty_phonetic_complement':    (0.00, 0.40),
    'penalty_phonogram_nonfinal':     (0.00, 0.60),
    'penalty_no_context_mult':        (0.20, 1.00),
    'context_strength_boost':         (0.00, 0.40),

    # Penalidades para sinais recém-adicionados
    'penalty_recently_critical': (0.30, 1.00),
    'penalty_recently_minor':    (0.50, 1.00),
    'penalty_recently_other':    (0.70, 1.00),

    # Penalidade contextual em _score_span
    'penalty_contextual_span':      (0.00, 0.60),
    'bonus_plural_span':            (0.00, 0.15),
    'bonus_location_span':          (0.00, 0.20),
    'bonus_location_modifier_span': (0.00, 0.15),
    'bonus_strong_det_span':        (0.00, 0.10),

    # Limiar de contexto padrão
    'thresh_default_context': (0.50, 0.99),
}

# Limiares por signo (otimizados individualmente)
# Bounds genéricos (0.50–0.99) — a otimização explora a faixa completa.
# Os valores default v1.28 ficam guardados em CONTEXT_THRESHOLDS (definido
# na FASE 2 do arquivo) e são lidos pelo _default_params() em tempo de execução.
CONTEXT_SIGNS_TO_OPTIMIZE = [
    'W24', 'W22', 'W23', 'R8', 'R7', 'D4', 'F4', 'K1', 'X1',
    'I9', 'I10', 'L1', 'L2', 'I5', 'I6', 'D58', 'D56', 'U1', 'U6',
]
for _s in CONTEXT_SIGNS_TO_OPTIMIZE:
    PARAM_SPACE[f'thresh_{_s}'] = (0.50, 0.99)


def _default_params() -> Dict[str, float]:
    """Retorna os parâmetros padrão (valores hardcoded da v1.28)."""
    # Limiares v1.28 hardcoded aqui para evitar dependência de CONTEXT_THRESHOLDS,
    # que é definido na FASE 2 (depois desta função).
    _DEFAULT_THRESHOLDS: Dict[str, float] = {
        'W24': 0.75, 'W22': 0.75, 'W23': 0.75,
        'R8':  0.75, 'R7':  0.75,
        'D4':  0.85, 'F4':  0.85,
        'K1':  0.80, 'X1':  0.70,
        'I9':  0.90, 'I10': 0.90, 'L1': 0.85,
        'L2':  0.80, 'I5':  0.80, 'I6': 0.80,
        'D58': 0.82, 'D56': 0.80,
        'U1':  0.78, 'U6':  0.78,
    }
    p: Dict[str, float] = {
        'w_sem': 0.38, 'w_struct': 0.34, 'w_rule': 0.28,
        'bonus_2src': 0.08, 'bonus_3src': 0.06,
        'bonus_final': 0.10, 'bonus_global_final': 0.06,
        'bonus_near_end': 0.06, 'bonus_group_end': 0.04,
        'bonus_reliable': 0.12,
        'bonus_plural_marker': 0.05, 'bonus_plural_head': 0.07,
        'bonus_strong_det': 0.08,
        'penalty_phonogram_primary_mult': 0.40,
        'penalty_phonetic_complement': 0.15,
        'penalty_phonogram_nonfinal': 0.35,
        'penalty_no_context_mult': 0.60,
        'context_strength_boost': 0.20,
        'penalty_recently_critical': 0.60,
        'penalty_recently_minor': 0.75,
        'penalty_recently_other': 0.85,
        'penalty_contextual_span': 0.25,
        'bonus_plural_span': 0.05,
        'bonus_location_span': 0.06,
        'bonus_location_modifier_span': 0.04,
        'bonus_strong_det_span': 0.03,
        'thresh_default_context': 0.75,
    }
    for s in CONTEXT_SIGNS_TO_OPTIMIZE:
        p[f'thresh_{s}'] = _DEFAULT_THRESHOLDS.get(s, 0.75)
    return p


def _flat_params() -> Dict[str, float]:
    """
    Option B — Flat uninformative prior.

    Philosophy: start knowing nothing. Every sign, source, position and
    category gets equal treatment. The training data must justify any
    differentiation from this neutral baseline.

    Initialisation rules:
      • Source weights (rule path): w_rule = 1.0  (only source in inner CV)
      • Source weights (LLM path):  w_sem_llm = w_struct_llm = w_rule_llm = 0.33
      • Bonuses: 0.0  — no position or category is preferred a priori
      • Multiplicative penalties: 1.0  — no sign is penalised a priori
      • Additive penalties: 0.0
      • Thresholds: 0.75  — midpoint, no sign is more context-dependent than another

    This is Option B + D combined:
      B → equal weights everywhere (flat prior)
      D → separate LLM source weights, also initialised flat
    """
    p: Dict[str, float] = {
        # Rule-path source weight (only weight active during inner CV)
        'w_sem':    0.0,   # legacy name — used as w_sem_llm reference
        'w_struct': 0.0,   # legacy name — used as w_struct_llm reference
        'w_rule':   1.00,   # sole source in inner CV → full weight

        # Option D: LLM-specific weights — flat equal prior
        'w_sem_llm':    0.0,
        'w_struct_llm': 0.0,
        'w_rule_llm':   1.0,

        # Bonuses: all zero — no a priori preference
        'bonus_2src':                0.00,
        'bonus_3src':                0.00,
        'bonus_final':               0.00,
        'bonus_global_final':        0.00,
        'bonus_near_end':            0.00,
        'bonus_group_end':           0.00,
        'bonus_reliable':            0.00,
        'bonus_plural_marker':       0.00,
        'bonus_plural_head':         0.00,
        'bonus_strong_det':          0.00,
        'bonus_plural_span':         0.00,
        'bonus_location_span':       0.00,
        'bonus_location_modifier_span': 0.00,
        'bonus_strong_det_span':     0.00,

        # Multiplicative penalties: 1.0 — neutral (no penalty)
        'penalty_phonogram_primary_mult': 1.00,
        'penalty_no_context_mult':        1.00,
        'penalty_recently_critical':      1.00,
        'penalty_recently_minor':         1.00,
        'penalty_recently_other':         1.00,

        # Additive penalties: 0.0 — neutral
        'penalty_phonetic_complement': 0.00,
        'penalty_phonogram_nonfinal':  0.00,
        'penalty_contextual_span':     0.00,
        'context_strength_boost':      0.00,

        # Thresholds: midpoint — no sign is more context-dependent than another
        'thresh_default_context': 0.75,
    }
    for s in CONTEXT_SIGNS_TO_OPTIMIZE:
        p[f'thresh_{s}'] = 0.75
    return p


def _init_params(strategy: str = 'flat') -> Dict[str, float]:
    """
    Returns initial parameters according to INIT_STRATEGY.
      'flat'    → Option B: uninformative flat prior (recommended)
      'default' → v1.28 expert-tuned values (original behaviour)
      'random'  → uniform random draw from PARAM_SPACE bounds
    """
    import random as _rand
    if strategy == 'flat':
        return _flat_params()
    elif strategy == 'random':
        p = {}
        for name, (lo, hi) in PARAM_SPACE.items():
            p[name] = lo + _rand.random() * (hi - lo)
        # Add LLM weights not in PARAM_SPACE
        p['w_sem_llm']    = 0.0
        p['w_struct_llm'] = 0.0
        p['w_rule_llm']   = 1.0
        return p
    else:  # 'default'
        return _default_params()


def _gather_rule_evidence(rule_result: Dict, tokens: List[str],
                           mdc_struct: Dict) -> Dict[tuple, Dict]:
    """
    Versão simplificada de _gather_candidate_evidence usando apenas rule_result.
    Usada durante a otimização (sem LLM) para velocidade máxima.
    """
    block_final  = set(mdc_struct.get('block_final_token_indices', []) or [])
    global_final = mdc_struct.get('final_sign_token_index', None)
    near_end     = set(mdc_struct.get('last_sign_token_indices', []) or [])
    plural_marks = set(mdc_struct.get('plural_marker_indices',  []) or [])
    plural_heads = set(mdc_struct.get('plural_head_indices',    []) or [])
    group_ends   = set(
        int(g['end_sign_token_index'])
        for g in (mdc_struct.get('groups') or [])
        if isinstance(g, dict) and g.get('end_sign_token_index') is not None
    )
    token_to_block = mdc_struct.get('token_index_to_block', {}) or {}

    pool: Dict[tuple, Dict] = {}
    for d in (rule_result.get('determinatives', []) or []):
        tok = (d.get('sign') or '').strip()
        idx = d.get('position')
        try:    idx = int(idx)
        except: continue
        if not tok or not (0 <= idx < len(tokens)):
            continue
        if tokens[idx] != tok or tok in MDL_SEPARATORS:
            continue
        conf_rule = {'high': 1.0, 'medium': 0.7}.get(
            str(d.get('confidence', '')).lower(), 0.4)
        key = (tok, idx)
        pool[key] = {
            'token': tok, 'index': idx,
            'sources': {'rules'},
            'conf_sem': 0.0, 'conf_struct': 0.0, 'conf_rule': conf_rule,
            'is_final':        idx in block_final,
            'is_global_final': (global_final is not None and idx == int(global_final)),
            'is_group_end':    idx in group_ends,
            'is_near_end':     idx in near_end,
            'is_plural_marker':idx in plural_marks,
            'is_plural_head':  idx in plural_heads,
            'is_phonetic_complement_likely': False,
            'block_id':        int(token_to_block.get(idx, -1)),
            'sequence_tokens': tokens,
            'semantic_hints':  '',
        }
    return pool


def _score_candidate_p(item: Dict, p: Dict[str, float]) -> float:
    """
    Versão parametrizada de _score_candidate.
    Aceita dict de parâmetros p em vez de constantes globais.
    """
    tok            = item.get('token', '')
    is_final       = bool(item.get('is_final', False))
    is_global_final= bool(item.get('is_global_final', False))
    is_group_end   = bool(item.get('is_group_end', False))
    is_near_end    = bool(item.get('is_near_end', False))
    is_plural_mark = bool(item.get('is_plural_marker', False))
    is_plural_head = bool(item.get('is_plural_head', False))
    sources        = item.get('sources') or set()
    src_n          = len(sources) if isinstance(sources, set) else 0

    # Bloqueio absoluto: fonogramas estritos
    if tok in SIGN_CATEGORIES.get('phonogram_strict', set()):
        return 0.0

    conf_sem    = float(item.get('conf_sem',    0.0))
    conf_struct = float(item.get('conf_struct', 0.0))
    conf_rule   = float(item.get('conf_rule',   0.0))

    # Option D: detect which path is active and use the appropriate weights
    has_llm_evidence = conf_sem > 0.0 or conf_struct > 0.0
    if has_llm_evidence:
        # LLM inference path — use LLM-specific source weights
        w_s = p.get('w_sem_llm',    p.get('w_sem',    0.0))
        w_t = p.get('w_struct_llm', p.get('w_struct', 0.0))
        w_r = p.get('w_rule_llm',   p.get('w_rule',   1.0))
    else:
        # Rule-only path (inner CV) — use single rule weight
        w_s = 0.0
        w_t = 0.0
        w_r = p['w_rule']

    score = w_s * conf_sem + w_t * conf_struct + w_r * conf_rule

    if src_n >= 2: score += p['bonus_2src']
    if src_n >= 3: score += p['bonus_3src']

    # Fonogramas primários
    if tok in SIGN_CATEGORIES.get('phonogram_primary', set()):
        score *= p['penalty_phonogram_primary_mult']

    # Sinais críticos de contexto
    is_context_critical = any(
        tok in SIGN_CATEGORIES.get(cat, set())
        for cat in ['context_vessel', 'context_divine', 'context_vision',
                    'context_bovine', 'context_aquatic', 'context_food',
                    'context_animal', 'context_insect', 'context_tool',
                    'context_body_action']
    )
    if is_context_critical:
        seq_toks = item.get('sequence_tokens', [])
        sem_hint = item.get('semantic_hints', '')
        has_ctx, ctx_str = validate_semantic_context(tok, seq_toks, sem_hint)
        threshold = p.get(f'thresh_{tok}', p['thresh_default_context'])
        if not has_ctx:
            if score < threshold:
                return 0.0
            score *= p['penalty_no_context_mult']
        else:
            score *= (1.0 + ctx_str * p['context_strength_boost'])

    # Penalidade para sinais recém-adicionados
    if tok in RECENTLY_ADDED_SIGNS:
        if tok in {'I9', 'I10', 'L1'}:
            score *= p['penalty_recently_critical']
        elif tok in {'D58', 'U1', 'L2', 'I5', 'I6'}:
            score *= p['penalty_recently_minor']
        else:
            score *= p['penalty_recently_other']

    # Bônus por categoria
    if tok in SIGN_CATEGORIES.get('reliable', set()):
        score += p['bonus_reliable']
    if tok in SIGN_CATEGORIES.get('plural_functional', set()):
        if is_plural_mark or is_near_end: score += p['bonus_plural_marker']
        if is_plural_head:                score += p['bonus_plural_head']

    # Bônus posicionais
    if is_final:                        score += p['bonus_final']
    if is_group_end and not is_final:   score += p['bonus_group_end']
    if is_global_final:                 score += p['bonus_global_final']
    if is_near_end and not is_final:    score += p['bonus_near_end']

    if (tok in STRONG_DETERMINATIVE_SIGNS
            and tok not in SIGN_CATEGORIES.get('reliable', set())):
        score += p['bonus_strong_det']

    # Penalidade complemento fonético
    if bool(item.get('is_phonetic_complement_likely', False)):
        score -= p['penalty_phonetic_complement'] if src_n <= 1 \
                 else p['penalty_phonetic_complement'] * 0.4

    # Penalidade fonograma primário fora do final
    if tok in SIGN_CATEGORIES.get('phonogram_primary', set()) and not is_final:
        score -= p['penalty_phonogram_nonfinal'] if src_n <= 1 \
                 else p['penalty_phonogram_nonfinal'] * 0.5

    return max(0.0, min(1.0, score))


def _score_span_p(span_indices: List[int], tokens: List[str],
                  occ_score: Dict[tuple, float], p: Dict[str, float]) -> float:
    """Versão parametrizada de _score_span."""
    if not span_indices:
        return 0.0
    span_toks = [tokens[i] for i in span_indices if 0 <= i < len(tokens)]
    s = sum(occ_score.get((tokens[i], i), 0.0)
            + (p['bonus_strong_det_span'] if tokens[i] in STRONG_DETERMINATIVE_SIGNS else 0.0)
            for i in span_indices if 0 <= i < len(tokens))
    if any(t in PLURAL_MARKERS        for t in span_toks): s += p['bonus_plural_span']
    if any(_is_final_location(t)      for t in span_toks): s += p['bonus_location_span']
    if (any(_is_location_modifier(t)  for t in span_toks)
            and any(_is_final_location(t) for t in span_toks)):
        s += p['bonus_location_modifier_span']
    if any(t in CONTEXTUAL_ONLY_SIGNS for t in span_toks): s -= p['penalty_contextual_span']
    return max(0.0, min(3.0, s))


def _select_dets_parametric(sample_data: Dict, p: Dict[str, float]) -> List[str]:
    """
    Seleciona determinativos usando scoring parametrizado.
    Retorna lista de sign codes (com repetições).
    """
    tokens     = sample_data['tokens']
    words      = sample_data['words']
    span_cands = sample_data['word_span_candidates']
    pool       = sample_data['rule_evidence_pool']

    # Score por ocorrência
    occ_score: Dict[tuple, float] = {
        key: _score_candidate_p(item, p) for key, item in pool.items()
    }

    # Beam search por palavra (versão simplificada)
    BEAM = 6
    beam = [{'score': 0.0, 'dets': []}]

    for wdata in span_cands:
        if not isinstance(wdata, dict):
            continue
        wid   = int(wdata.get('word_id', 0))
        cands = wdata.get('candidates') or []
        new_beam = []
        for entry in beam:
            for c in cands:
                sp = [int(i) for i in (c.get('span_token_indices') or [])]
                # Validar span
                if any(not (0 <= i < len(tokens)) or tokens[i] in MDL_SEPARATORS
                       or tokens[i] in NEVER_DETERMINATIVE_SIGNS for i in sp):
                    sp = []
                sp_score = _score_span_p(sp, tokens, occ_score, p)
                dets = [tokens[i] for i in sp]
                new_beam.append({
                    'score': entry['score'] + sp_score,
                    'dets':  entry['dets'] + dets,
                })
        new_beam.sort(key=lambda x: -x['score'])
        beam = new_beam[:BEAM]

    return beam[0]['dets'] if beam else []


def _precompute_loo_samples(loo_df: 'pd.DataFrame',
                             gt_dict: Dict[str, List[str]]) -> List[Dict]:
    """
    Pré-computa tokenização, estrutura e evidências de regras para cada
    amostra LOO. Chamado UMA VEZ antes da otimização.
    """
    import re as _re
    samples = []
    for _, row in loo_df.iterrows():
        seq = str(row.get('Manuel_de_Codage', row.get('Code', ''))).strip()
        if not seq:
            continue
        norm_seq = _re.sub(r'\s+', '', seq)
        tokens   = tokenize_mdc(seq)
        mdc_st   = build_mdc_structure(tokens)
        words    = build_word_boundaries(tokens, mdc_st)
        spans    = generate_suffix_span_candidates(words, tokens, max_len=6)
        rule_res = analyze_rule_based(tokens)
        pool     = _gather_rule_evidence(rule_res, tokens, mdc_st)
        gt_signs = gt_dict.get(norm_seq, [])
        samples.append({
            'sequence':            seq,
            'tokens':              tokens,
            'mdc_structure':       mdc_st,
            'words':               words,
            'word_span_candidates': spans,
            'rule_result':         rule_res,
            'rule_evidence_pool':  pool,
            'gt_signs':            gt_signs,
        })
    return samples


def _run_loo_fast(precomputed: List[Dict], p: Dict[str, float]) -> float:
    """
    Executa LOO completo com parâmetros p sobre dados pré-computados.
    Retorna micro-F1 (0..1). Cada chamada leva ~milissegundos.
    """
    tp = fp = fn = 0
    for s in precomputed:
        predicted = _select_dets_parametric(s, p)
        m = evaluate_single_prediction(predicted, s['gt_signs'])
        tp += m['tp']; fp += m['fp']; fn += m['fn']
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def optimize_parameters_loo(precomputed: List[Dict],
                             n_trials: int = 200,
                             cache_path: str = None) -> Dict[str, float]:
    """
    Otimiza os parâmetros de scoring usando Optuna (TPE Bayesiano) com
    LOO micro-F1 como função objetivo.

    Args:
        precomputed : saída de _precompute_loo_samples()
        n_trials    : número de tentativas do Optuna
        cache_path  : se fornecido, salva/carrega resultado em JSON

    Returns:
        Dicionário com os melhores parâmetros encontrados.
    """
    # ── Tentar carregar cache ─────────────────────────────────────────────
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as _f:
                cached = json.load(_f)
            if isinstance(cached.get('best_params'), dict):
                print(f"✅ Parâmetros carregados do cache: {cache_path}")
                print(f"   (F1 cached = {cached.get('best_f1', '?')*100:.2f}%)")
                return cached['best_params']
        except Exception as _e:
            print(f"⚠️  Cache inválido ({_e}), re-otimizando...")

    if not _HAS_OPTUNA:
        print("⚠️  optuna não instalado. Execute: pip install optuna")
        print("   Usando parâmetros padrão.")
        return _default_params()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Baseline com parâmetros padrão ───────────────────────────────────
    default_p  = _default_params()
    baseline_f1 = _run_loo_fast(precomputed, default_p)
    print(f"📊 Baseline LOO micro-F1 (parâmetros v1.28): {baseline_f1*100:.2f}%")

    # ── Função objetivo do Optuna ─────────────────────────────────────────
    def _objective(trial: Any) -> float:
        p: Dict[str, float] = {}
        for name, (lo, hi) in PARAM_SPACE.items():
            p[name] = trial.suggest_float(name, lo, hi)
        return _run_loo_fast(precomputed, p)

    # ── Criar study e adicionar o ponto baseline como warm-start ─────────
    sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    # Warm-start: injetar parâmetros default como primeira tentativa
    study.enqueue_trial(
        {name: default_p.get(name, (lo + hi) / 2)
         for name, (lo, hi) in PARAM_SPACE.items()}
    )

    # ── Callback de progresso ─────────────────────────────────────────────
    _best = [baseline_f1]
    def _progress_cb(study, trial):
        if study.best_value > _best[0]:
            _best[0] = study.best_value
            print(f"  🎯 Trial {trial.number:>4}: F1={study.best_value*100:.3f}%  "
                  f"(+{(study.best_value - baseline_f1)*100:.2f}pp vs baseline)")

    print(f"\n🔍 Iniciando otimização Optuna ({n_trials} trials)...")
    print(f"   Parâmetros no espaço: {len(PARAM_SPACE)}")
    print(f"   Amostras LOO por avaliação: {len(precomputed)}")

    study.optimize(_objective, n_trials=n_trials, callbacks=[_progress_cb])

    best_p  = study.best_params
    best_f1 = study.best_value

    print(f"\n✅ Otimização concluída!")
    print(f"   Baseline F1 : {baseline_f1*100:.2f}%")
    print(f"   Melhor F1   : {best_f1*100:.2f}%")
    print(f"   Ganho       : {(best_f1 - baseline_f1)*100:+.2f}pp")

    # ── Salvar cache ──────────────────────────────────────────────────────
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as _f:
                json.dump({'best_params': best_p, 'best_f1': best_f1,
                           'baseline_f1': baseline_f1,
                           'n_trials': n_trials,
                           'n_samples': len(precomputed)},
                          _f, indent=2, ensure_ascii=False)
            print(f"✅ Parâmetros salvos em: {cache_path}")
        except Exception as _e:
            print(f"⚠️  Não foi possível salvar cache: {_e}")

    return best_p


def apply_optimized_params(p: Dict[str, float]) -> None:
    """
    Applies optimised parameters globally.

    Option D: stores both rule-path and LLM-path source weights so that
    _score_candidate_p can select the correct set at runtime depending on
    whether LLM evidence is present (conf_sem > 0 or conf_struct > 0).
    """
    global CONTEXT_THRESHOLDS, _ACTIVE_PARAMS

    for sign in CONTEXT_SIGNS_TO_OPTIMIZE:
        key = f'thresh_{sign}'
        if key in p:
            CONTEXT_THRESHOLDS[sign] = p[key]

    # Ensure LLM weights are present (carry flat prior if missing)
    full_p = dict(p)
    full_p.setdefault('w_sem_llm',    LLM_SOURCE_WEIGHTS_INIT.get('w_sem_llm',    0.1))
    full_p.setdefault('w_struct_llm', LLM_SOURCE_WEIGHTS_INIT.get('w_struct_llm', 0.1))
    full_p.setdefault('w_rule_llm',   LLM_SOURCE_WEIGHTS_INIT.get('w_rule_llm',   0.8))

    _ACTIVE_PARAMS = full_p
    print(f"✅ Parâmetros aplicados  |  "
          f"LLM weights: sem={full_p['w_sem_llm']:.2f}  "
          f"struct={full_p['w_struct_llm']:.2f}  "
          f"rule={full_p['w_rule_llm']:.2f}")


def _apply_params_to_score_candidate() -> None:
    """
    Substitui a função global _score_candidate pela versão parametrizada
    usando _ACTIVE_PARAMS. Chamado após apply_optimized_params().
    """
    import sys
    _mod = sys.modules[__name__]

    def _patched_score_candidate(item: Dict[str, Any]) -> float:
        return _score_candidate_p(item, _ACTIVE_PARAMS)

    _mod._score_candidate = _patched_score_candidate


def _plot_optimization_results(study: Any,
                                baseline_f1: float,
                                output_dir: str) -> None:
    """Gera gráficos de convergência e importância de parâmetros."""
    if not _HAS_OPTUNA:
        print("⚠️  optuna não disponível — gráficos de otimização ignorados.")
        return
    try:
        # 1. Gráfico de convergência
        trials_f1 = [t.value for t in study.trials if t.value is not None]
        best_so_far = [max(trials_f1[:i+1]) for i in range(len(trials_f1))]
        plt.figure(figsize=(10, 5))
        plt.plot(trials_f1,    color='steelblue', alpha=0.4, label='F1 por trial')
        plt.plot(best_so_far,  color='red',       linewidth=2, label='Melhor F1 acumulado')
        plt.axhline(baseline_f1, color='orange', linestyle='--',
                    label=f'Baseline v1.28 ({baseline_f1*100:.1f}%)')
        plt.xlabel('Trial', fontsize=12)
        plt.ylabel('LOO Micro-F1', fontsize=12)
        plt.title('Convergência da Otimização — Optuna LOO (v1.30)', fontsize=13, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        conv_path = f"{output_dir}/opt_convergence.png"
        plt.savefig(conv_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Gráfico convergência: {conv_path}")

        # 2. Importância dos parâmetros
        try:
            importances = optuna.importance.get_param_importances(study)
            top_params  = list(importances.items())[:15]
            names  = [k for k, _ in top_params]
            values = [v for _, v in top_params]
            plt.figure(figsize=(10, 7))
            plt.barh(names[::-1], values[::-1], color='steelblue', alpha=0.85)
            plt.xlabel('Importância Relativa', fontsize=11)
            plt.title('Importância dos Parâmetros — LOO Optuna (v1.30)',
                      fontsize=13, fontweight='bold')
            plt.tight_layout()
            imp_path = f"{output_dir}/opt_param_importance.png"
            plt.savefig(imp_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Gráfico importância: {imp_path}")
        except Exception as _ie:
            print(f"⚠️  Importância de parâmetros indisponível: {_ie}")

    except Exception as _e:
        print(f"⚠️  Erro nos gráficos de otimização: {_e}")


def print_param_comparison(default_p: Dict, best_p: Dict) -> None:
    """Imprime tabela comparando parâmetros default (v1.28), flat prior (B) e optimizados."""
    flat_p = _flat_params()
    print("\n" + "═"*80)
    print("⚙️   COMPARAÇÃO DE PARÂMETROS — v1.28 Default | Flat Prior (B) | Optimizado (B+D)")
    print("═"*80)
    print(f"  {'Parâmetro':<38} {'Default':>9} {'Flat(B)':>9} {'Optimized':>10} {'Δ opt-flat':>11}")
    print("  " + "-"*78)
    all_keys = sorted(set(default_p) | set(flat_p) | set(best_p))
    for k in all_keys:
        d  = default_p.get(k, float('nan'))
        f  = flat_p.get(k, float('nan'))
        o  = best_p.get(k, float('nan'))
        try:
            delta = o - f
            flag = '  ⬆' if delta > 0.02 else ('  ⬇' if delta < -0.02 else '')
            print(f"  {k:<38} {d:>9.3f} {f:>9.3f} {o:>10.3f} {delta:>+10.3f}{flag}")
        except Exception:
            print(f"  {k:<38} {'?':>9} {'?':>9} {'?':>10}")

    # Highlight the LLM weights separately (Option D)
    print("\n  ── Option D: LLM source weights ──────────────────────────────────────────")
    for k in ['w_sem_llm', 'w_struct_llm', 'w_rule_llm']:
        f = flat_p.get(k, 0.5)
        o = best_p.get(k, 0.5)
        print(f"  {k:<38} {'n/a':>9} {f:>9.3f} {o:>10.3f}  (not optimised by inner CV)")
    print("═"*80 + "\n")


# ============================================================================
# V1.32 - NESTED CV: 5 × (5×2 CV) + TEST SET
# ============================================================================
#
# DESIGN:
#
#   ┌─ Outer loop (5 splits, each with a different 80/19 partition) ──────────┐
#   │                                                                          │
#   │  For each outer split r ∈ {1..5}:                                        │
#   │    train_val = 80 samples   test = 19 samples                            │
#   │                                                                          │
#   │    ┌─ Inner 5×2 CV (fast, rule-based only) ──────────────────────────┐  │
#   │    │  For each inner rep k ∈ {1..5}:                                  │  │
#   │    │    Shuffle 80 → split A(40) / B(40)                              │  │
#   │    │    Optimize params on A  →  validate on B  → val_f1_AB, p_AB     │  │
#   │    │    Optimize params on B  →  validate on A  → val_f1_BA, p_BA     │  │
#   │    │  10 (param_set, val_f1) pairs total                               │  │
#   │    │  best_params = param set with highest val_f1                      │  │
#   │    └──────────────────────────────────────────────────────────────────┘  │
#   │                                                                          │
#   │    Evaluate 19 test samples with LLM + best_params                       │
#   │    Few-shot examples drawn from the 80 train_val samples                 │
#   │    Record: test P/R/F1, null_stats, per-sign metrics                     │
#   │                                                                          │
#   └──────────────────────────────────────────────────────────────────────────┘
#
#   Final report: mean ± std across 5 outer folds for all metrics
#   Generalization gap: val_f1 vs test_f1
#
# TOTAL LLM CALLS: 5 outer × 19 test = 95  (vs 99 for full LOO)
# ============================================================================

def _make_outer_splits(df_loo: 'pd.DataFrame',
                       n_splits: int = 5,
                       random_state: int = 42) -> List[tuple]:
    """
    Gera n_splits partições 80/19 com test sets NÃO sobrepostos.

    Estratégia: shuffle único dos 99 índices, depois divide em 5 janelas
    consecutivas de 19 ou 20 amostras. Cada janela é o test set de uma
    repetição; o resto (80) é o train/val.

    Com 99 amostras e n_splits=5:
      - 5×19 = 95 → as primeiras 4 janelas têm 20, a última tem 19
        (ou distribui 4×20 + 1×19 conforme n % n_splits)
      - Cada amostra aparece em EXACTAMENTE 1 test set
      - Cada test set tem entre 19 e 20 amostras (≈19 como pedido)

    Para forçar exactamente 19 em todos os folds usa-se as primeiras
    99 - (99 % n_splits) = 95 amostras após shuffle, descartando 4.
    Aqui optamos por distribuição natural (4×20 + 1×19) para não
    descartar dados.
    """
    import random as _rand

    n = len(df_loo)                         # 99
    indices = list(range(n))

    rng = _rand.Random(random_state)
    rng.shuffle(indices)                    # um único shuffle reprodutível

    # Dividir em n_splits janelas o mais iguais possível
    # Com 99 e n_splits=5 → tamanhos [20, 20, 20, 20, 19]
    fold_size  = n // n_splits              # 19
    remainder  = n % n_splits              # 4  (os 4 folds extra têm +1)

    splits = []
    start  = 0
    for i in range(n_splits):
        sz        = fold_size + (1 if i < remainder else 0)
        test_idx  = indices[start : start + sz]
        train_val = indices[:start] + indices[start + sz:]
        splits.append((train_val, test_idx))
        start    += sz

    return splits


def _optimize_on_samples(samples: List[Dict],
                          n_trials: int = 100,
                          seed: int = 0) -> tuple:
    """
    Roda Optuna num subconjunto de amostras pré-computadas.
    Retorna (best_params, best_f1, baseline_f1, init_f1, n_improvements).

    init_f1       — F1 do ponto de partida (flat/default/random)
    n_improvements — quantos dos n_trials melhoraram o ponto de partida
    """
    default_p  = _default_params()
    init_p     = _init_params(INIT_STRATEGY)
    baseline   = _run_loo_fast(samples, default_p)  # sempre v1.28
    init_f1    = _run_loo_fast(samples, init_p)     # F1 do prior escolhido

    if not _HAS_OPTUNA:
        return init_p, init_f1, baseline, init_f1, 0

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_improvements = 0

    def _obj(trial):
        nonlocal n_improvements
        p = {name: trial.suggest_float(name, lo, hi)
             for name, (lo, hi) in PARAM_SPACE.items()}
        p.setdefault('w_sem_llm',    init_p.get('w_sem_llm',    0.1))
        p.setdefault('w_struct_llm', init_p.get('w_struct_llm', 0.1))
        p.setdefault('w_rule_llm',   init_p.get('w_rule_llm',   0.8))
        f1 = _run_loo_fast(samples, p)
        if f1 > init_f1:
            n_improvements += 1
        return f1

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    warm    = {name: init_p.get(name, (lo+hi)/2)
               for name, (lo, hi) in PARAM_SPACE.items()}
    study.enqueue_trial(warm)
    study.optimize(_obj, n_trials=n_trials)

    best = study.best_params
    best['w_sem_llm']    = init_p.get('w_sem_llm',    0.1)
    best['w_struct_llm'] = init_p.get('w_struct_llm', 0.1)
    best['w_rule_llm']   = init_p.get('w_rule_llm',   0.8)

    return best, study.best_value, baseline, init_f1, n_improvements


def _run_5x2_inner_cv(train_val_samples: List[Dict],
                       n_reps: int = 5,
                       n_trials: int = 100,
                       random_state: int = 42) -> Dict:
    """
    Executa a 5×2 cross-validation interna nos train_val_samples (≈80).

    Para cada repetição k:
      - Shuffle e split 50/50 → fold_A (40), fold_B (40)
      - Otimiza params em A, valida em B  → (p_AB, val_f1_AB)
      - Otimiza params em B, valida em A  → (p_BA, val_f1_BA)

    Devolve:
      - inner_records: lista de 2×n_reps dicts com params e val_f1
      - best_params: o conjunto de params com maior val_f1
      - val_f1_mean ± val_f1_std (sobre as 10 avaliações)
    """
    import random as _rand
    import numpy as np

    rng = _rand.Random(random_state)
    n   = len(train_val_samples)
    inner_records = []

    for rep in range(n_reps):
        shuffled = train_val_samples[:]
        rng.shuffle(shuffled)
        mid = n // 2
        fold_A, fold_B = shuffled[:mid], shuffled[mid:]

        for train_fold, val_fold, label in [
                (fold_A, fold_B, f'rep{rep+1}_AB'),
                (fold_B, fold_A, f'rep{rep+1}_BA')]:

            opt_params, opt_f1, baseline, init_f1, n_impr = _optimize_on_samples(
                train_fold, n_trials=n_trials, seed=rep * 100 + len(label))
            val_f1 = _run_loo_fast(val_fold, opt_params)

            inner_records.append({
                'label':      label,
                'opt_f1':     opt_f1,
                'val_f1':     val_f1,
                'baseline':   baseline,
                'init_f1':    init_f1,
                'n_improvements': n_impr,
                'params':     opt_params,
                'n_train':    len(train_fold),
                'n_val':      len(val_fold),
            })
            gain = opt_f1 - init_f1
            print(f"    [{label}] "
                  f"init({INIT_STRATEGY})={init_f1*100:.1f}%  "
                  f"opt_F1={opt_f1*100:.1f}%  "
                  f"(Δ={gain*100:+.1f}pp, {n_impr}/{n_trials} trials improved)  "
                  f"val_F1={val_f1*100:.1f}%  "
                  f"baseline(v1.28)={baseline*100:.1f}%")

    val_f1s   = [r['val_f1'] for r in inner_records]
    best_rec  = max(inner_records, key=lambda r: r['val_f1'])
    best_params = best_rec['params']

    return {
        'inner_records': inner_records,
        'best_params':   best_params,
        'best_val_f1':   best_rec['val_f1'],
        'val_f1_mean':   float(np.mean(val_f1s)),
        'val_f1_std':    float(np.std(val_f1s)),
        'val_f1_min':    float(np.min(val_f1s)),
        'val_f1_max':    float(np.max(val_f1s)),
    }


def _evaluate_test_with_llm(test_df: 'pd.DataFrame',
                              train_val_df: 'pd.DataFrame',
                              gt_dict: Dict[str, List[str]],
                              best_params: Dict[str, float],
                              n_few_shot: int = 5,
                              outer_fold: int = 0) -> List[Dict]:
    """
    Avalia as ~19 amostras de teste usando o pipeline LLM completo.

    best_params são aplicados globalmente antes das chamadas LLM.
    Os exemplos few-shot são selecionados dos 80 train_val samples.

    Retorna lista de dicts compatíveis com compute_loo_metrics.
    """
    import re as _re

    # Aplicar parâmetros otimizados
    apply_optimized_params(best_params)
    _apply_params_to_score_candidate()

    few_shot_examples = select_expert_examples_loo(train_val_df, n_examples=n_few_shot,
                                                    test_sequence=None)  # global pool, updated per sample below

    test_results = []
    for test_i, (_, row) in enumerate(test_df.iterrows()):
        seq = str(row.get('Manuel_de_Codage', row.get('Code', ''))).strip()
        if not seq:
            continue
        img = row.get('image_file', f'test_fold{outer_fold}_{test_i}')
        norm_seq = _re.sub(r'\s+', '', seq)
        gt_signs = gt_dict.get(norm_seq, [])

        print(f"    [Outer {outer_fold+1} | Test {test_i+1}/{len(test_df)}] "
              f"{seq[:50]}")

        # V1.33: per-sample similarity-based few-shot selection
        few_shot_examples = select_expert_examples_loo(
            train_val_df, n_examples=n_few_shot, test_sequence=seq)

        if USE_LANGGRAPH:
            result = analyze_with_langgraph(seq, img, expert_examples=few_shot_examples,
                                            optimised_params=best_params)
            _dets  = result.get('determinatives', []) if result.get('success') else []
        else:
            tokens = tokenize_mdc(seq)
            _dets  = analyze_rule_based(tokens).get('determinatives', [])

        predicted_signs = [d['sign'] for d in _dets
                           if isinstance(d, dict) and d.get('sign')]
        metrics = evaluate_single_prediction(predicted_signs, gt_signs)

        test_results.append({
            'loo_index':       test_i,
            'outer_fold':      outer_fold,
            'sequence':        seq,
            'predicted_signs': predicted_signs,
            'gt_signs':        gt_signs,
            'metrics':         metrics,
            'few_shot_n':      len(few_shot_examples),
        })
        mm = metrics
        print(f"      Pred={predicted_signs}  GT={gt_signs}  "
              f"type={mm['outcome_type']}  F1={mm['f1']*100:.1f}%")

    return test_results


def run_nested_cv(df_loo: 'pd.DataFrame',
                  gt_dict: Dict[str, List[str]]) -> Dict:
    """
    Executa a avaliação completa 5 × (5×2 CV + Teste LLM).

    Pipeline por split externo:
      1. Divide df_loo em train_val (≈80) e test (≈19)
      2. Pré-computa evidências de regras para ambos os conjuntos
      3. Roda 5×2 CV interno nos 80 → encontra best_params
      4. Avalia os 19 com LLM + best_params
      5. Coleta métricas de validação e teste

    Retorna dict com resultados por fold externo + sumário agregado.
    """
    import numpy as np
    import re as _re

    print("\n" + "="*70)
    print("🔬 V1.32: NESTED CV — 5 × (5×2 CV) + TESTE LLM")
    print("="*70)
    print(f"  Dataset: {len(df_loo)} amostras  |  "
          f"Outer folds: {OUTER_N_SPLITS}  |  "
          f"Inner reps: {INNER_N_REPS}×2={INNER_N_REPS*2}")
    print(f"  Train/Val: ~{len(df_loo)*4//5} por fold  |  "
          f"Test: ~{len(df_loo)//OUTER_N_SPLITS} por fold")
    print(f"  Total LLM calls: ~{OUTER_N_SPLITS * (len(df_loo)//OUTER_N_SPLITS)}")

    outer_splits = _make_outer_splits(df_loo, OUTER_N_SPLITS, OUTER_RANDOM_STATE)
    outer_results = []

    for outer_i, (tv_idx, test_idx) in enumerate(outer_splits):
        print(f"\n{'─'*70}")
        print(f"OUTER FOLD {outer_i+1}/{OUTER_N_SPLITS}  "
              f"(train_val={len(tv_idx)}, test={len(test_idx)})")
        print(f"{'─'*70}")

        train_val_df = df_loo.iloc[tv_idx].reset_index(drop=True)
        test_df      = df_loo.iloc[test_idx].reset_index(drop=True)

        # ── Pré-computar evidências ────────────────────────────────────
        print(f"  📦 Pré-computando evidências de regras...")
        tv_samples   = _precompute_loo_samples(train_val_df, gt_dict)
        test_samples = _precompute_loo_samples(test_df,      gt_dict)
        print(f"     train_val: {len(tv_samples)} | test: {len(test_samples)}")

        # ── 5×2 Inner CV ──────────────────────────────────────────────
        print(f"\n  🔄 5×2 Inner CV ({INNER_N_REPS} reps × 2 halvos):")
        inner = _run_5x2_inner_cv(tv_samples, INNER_N_REPS, INNER_N_TRIALS,
                                   random_state=OUTER_RANDOM_STATE + outer_i * 13)
        print(f"  ✅ Val F1: {inner['val_f1_mean']*100:.2f}% ± "
              f"{inner['val_f1_std']*100:.2f}%  "
              f"[{inner['val_f1_min']*100:.1f}%–{inner['val_f1_max']*100:.1f}%]")
        print(f"     Best params set: val_F1={inner['best_val_f1']*100:.2f}%")

        # ── Fast rule-based test (reference, without LLM) ─────────────
        fast_test_results = []
        for s in test_samples:
            pred = _select_dets_parametric(s, inner['best_params'])
            m    = evaluate_single_prediction(pred, s['gt_signs'])
            fast_test_results.append({
                'loo_index': 0, 'outer_fold': outer_i,
                'sequence': s['sequence'],
                'predicted_signs': pred, 'gt_signs': s['gt_signs'],
                'metrics': m, 'few_shot_n': 0,
            })
        fast_metrics = compute_loo_metrics(fast_test_results)

        # ── LLM Test evaluation ────────────────────────────────────────
        print(f"\n  🤖 Avaliando {len(test_df)} amostras de teste com LLM...")
        llm_test_results = _evaluate_test_with_llm(
            test_df, train_val_df, gt_dict,
            inner['best_params'], TEST_N_FEW_SHOT, outer_i)
        llm_metrics = compute_loo_metrics(llm_test_results)

        # ── Rule-based baseline (default params) on test ───────────────
        default_fast = []
        for s in test_samples:
            pred = _select_dets_parametric(s, _default_params())
            m    = evaluate_single_prediction(pred, s['gt_signs'])
            default_fast.append({
                'loo_index': 0, 'outer_fold': outer_i,
                'sequence': s['sequence'],
                'predicted_signs': pred, 'gt_signs': s['gt_signs'],
                'metrics': m, 'few_shot_n': 0,
            })
        baseline_metrics = compute_loo_metrics(default_fast)

        fold_result = {
            'outer_fold':        outer_i,
            'n_train_val':       len(tv_idx),
            'n_test':            len(test_idx),
            'inner':             inner,
            'llm_test_metrics':  llm_metrics,
            'fast_test_metrics': fast_metrics,
            'baseline_metrics':  baseline_metrics,
            'llm_test_results':  llm_test_results,
        }
        outer_results.append(fold_result)

        # ── Summary for this fold ──────────────────────────────────────
        lm = llm_metrics; fm = fast_metrics; bm = baseline_metrics
        print(f"\n  📊 FOLD {outer_i+1} SUMMARY:")
        print(f"  {'Metric':<30} {'LLM Test':>10} {'Fast Test':>10} {'Baseline':>10}")
        print(f"  {'-'*62}")
        for label, metrics_src in [('Macro-F1 corrected', 'macro'),
                                    ('Micro-F1',           'micro'),
                                    ('Null Accuracy',      'null_stats')]:
            if label == 'Null Accuracy':
                lv = lm['null_stats'].get('null_accuracy')
                fv = fm['null_stats'].get('null_accuracy')
                bv = bm['null_stats'].get('null_accuracy')
                # V1.33: show N/A when no GT-empty sequences exist in this fold
                fmt = lambda x: f"{x*100:.1f}%" if x is not None else "N/A"
            else:
                lv = lm[metrics_src]['f1']
                fv = fm[metrics_src]['f1']
                bv = bm[metrics_src]['f1']
                fmt = lambda x: f"{x*100:.2f}%"
            print(f"  {label:<30} {fmt(lv):>10} {fmt(fv):>10} {fmt(bv):>10}")

    # ── Aggregate across outer folds ───────────────────────────────────────
    return _aggregate_nested_cv_results(outer_results)


def _aggregate_nested_cv_results(outer_results: List[Dict]) -> Dict:
    """
    Agrega os resultados dos 5 folds externos.
    Calcula mean ± std para métricas de validação interna e teste LLM.
    """
    import numpy as np

    def _collect(key_path):
        """Extrai um valor aninhado de cada fold result."""
        vals = []
        for r in outer_results:
            obj = r
            for k in key_path:
                obj = obj[k]
            vals.append(float(obj))
        return np.array(vals)

    metrics_to_collect = {
        'val_f1_mean':          ['inner', 'val_f1_mean'],
        'val_f1_std':           ['inner', 'val_f1_std'],
        'llm_macro_f1':         ['llm_test_metrics', 'macro', 'f1'],
        'llm_macro_p':          ['llm_test_metrics', 'macro', 'precision'],
        'llm_macro_r':          ['llm_test_metrics', 'macro', 'recall'],
        'llm_micro_f1':         ['llm_test_metrics', 'micro', 'f1'],
        'llm_macro_classic_f1': ['llm_test_metrics', 'macro_classic', 'f1'],
        'fast_macro_f1':        ['fast_test_metrics', 'macro', 'f1'],
        'fast_micro_f1':        ['fast_test_metrics', 'micro', 'f1'],
        'baseline_macro_f1':    ['baseline_metrics', 'macro', 'f1'],
        'baseline_micro_f1':    ['baseline_metrics', 'micro', 'f1'],
    }

    agg = {}
    for name, path in metrics_to_collect.items():
        try:
            vals = _collect(path)
            agg[name] = {'values': vals.tolist(),
                         'mean':   float(vals.mean()),
                         'std':    float(vals.std()),
                         'min':    float(vals.min()),
                         'max':    float(vals.max())}
        except Exception as e:
            agg[name] = {'error': str(e)}

    # Generalization gap: test_f1 - val_f1
    try:
        llm_vals = np.array(agg['llm_macro_f1']['values'])
        val_vals = np.array(agg['val_f1_mean']['values'])
        gap = llm_vals - val_vals
        agg['generalization_gap'] = {
            'values': gap.tolist(),
            'mean':   float(gap.mean()),
            'std':    float(gap.std()),
        }
    except Exception:
        pass

    return {'outer_results': outer_results, 'aggregate': agg,
            'n_outer_folds': len(outer_results)}


def print_nested_cv_report(nested_result: Dict) -> None:
    """Imprime relatório completo da avaliação Nested CV (v1.32)."""
    agg = nested_result['aggregate']
    n   = nested_result['n_outer_folds']
    import numpy as np

    print("\n" + "═"*70)
    print("📊  NESTED CV RESULTS — v1.32  (5 outer × 5×2 inner)")
    print("═"*70)

    def _fmt(key, pct=True):
        d = agg.get(key, {})
        if 'mean' not in d: return 'N/A'
        m, s = d['mean'], d['std']
        return f"{m*100:.2f}% ± {s*100:.2f}%" if pct else f"{m:.4f} ± {s:.4f}"

    print(f"\n{'INNER 5×2 CV (validação nos 80, sem LLM)':}")
    print(f"  Val macro-F1 mean ± std : {_fmt('val_f1_mean')}")

    print(f"\n{'TESTE LLM (~19 amostras por fold)':}")
    print(f"  Macro-F1 corrected ✅   : {_fmt('llm_macro_f1')}")
    print(f"  Macro-F1 classic        : {_fmt('llm_macro_classic_f1')}")
    print(f"  Micro-F1                : {_fmt('llm_micro_f1')}")
    print(f"  Macro-Precision         : {_fmt('llm_macro_p')}")
    print(f"  Macro-Recall            : {_fmt('llm_macro_r')}")

    print(f"\n{'TESTE RÁPIDO (parâmetros otimizados, sem LLM)':}")
    print(f"  Fast Macro-F1           : {_fmt('fast_macro_f1')}")
    print(f"  Fast Micro-F1           : {_fmt('fast_micro_f1')}")

    print(f"\n{'BASELINE (parâmetros v1.28, sem LLM)':}")
    print(f"  Baseline Macro-F1       : {_fmt('baseline_macro_f1')}")
    print(f"  Baseline Micro-F1       : {_fmt('baseline_micro_f1')}")

    gap = agg.get('generalization_gap', {})
    if 'mean' in gap:
        sign = '+' if gap['mean'] >= 0 else ''
        print(f"\n{'GENERALIZATION GAP (LLM test − val)':}")
        print(f"  Δ Macro-F1 mean ± std   : {sign}{gap['mean']*100:.2f}% "
              f"± {gap['std']*100:.2f}%")
        print(f"  (negativo → overfitting ao conjunto de val)")

    print(f"\n{'POR FOLD EXTERNO':}")
    print(f"  {'Fold':<6} {'n_test':>6} {'Val F1':>9} {'LLM F1':>9} "
          f"{'Fast F1':>9} {'Base F1':>9}")
    print("  " + "-"*52)
    for r in nested_result['outer_results']:
        oi   = r['outer_fold']
        ntst = r['n_test']
        vf1  = r['inner']['val_f1_mean']
        lf1  = r['llm_test_metrics']['macro']['f1']
        ff1  = r['fast_test_metrics']['macro']['f1']
        bf1  = r['baseline_metrics']['macro']['f1']
        print(f"  {oi+1:<6} {ntst:>6} {vf1*100:>8.2f}% {lf1*100:>8.2f}% "
              f"{ff1*100:>8.2f}% {bf1*100:>8.2f}%")
    print("═"*70 + "\n")


def export_nested_cv_results(nested_result: Dict, output_dir: str) -> None:
    """Exporta CSVs e JSON com os resultados detalhados da Nested CV."""
    import json as _json

    agg = nested_result['aggregate']

    # ── Per-fold summary CSV ───────────────────────────────────────────────
    fold_rows = []
    for r in nested_result['outer_results']:
        oi = r['outer_fold']
        inner = r['inner']
        lm = r['llm_test_metrics']
        fm = r['fast_test_metrics']
        bm = r['baseline_metrics']
        ns = lm.get('null_stats', {})
        fold_rows.append({
            'outer_fold':      oi + 1,
            'n_train_val':     r['n_train_val'],
            'n_test':          r['n_test'],
            'val_f1_mean':     round(inner['val_f1_mean'], 4),
            'val_f1_std':      round(inner['val_f1_std'],  4),
            'best_val_f1':     round(inner['best_val_f1'], 4),
            'llm_macro_f1':    round(lm['macro']['f1'],   4),
            'llm_micro_f1':    round(lm['micro']['f1'],   4),
            'llm_macro_p':     round(lm['macro']['precision'], 4),
            'llm_macro_r':     round(lm['macro']['recall'],    4),
            'llm_macro_classic_f1': round(lm.get('macro_classic',{}).get('f1',0), 4),
            'fast_macro_f1':   round(fm['macro']['f1'],   4),
            'baseline_macro_f1': round(bm['macro']['f1'], 4),
            'n_true_null':     ns.get('n_true_null', ''),
            'n_fp_null':       ns.get('n_fp_null',   ''),
            'n_fn_null':       ns.get('n_fn_null',   ''),
            'n_standard':      ns.get('n_standard',  ''),
            'null_accuracy':   round(ns['null_accuracy'],4) if ns.get('null_accuracy') else '',
            'fn_null_rate':    round(ns['fn_null_rate'],4)  if ns.get('fn_null_rate')  else '',
        })
    fold_csv_path = f"{output_dir}/nested_cv_per_fold.csv"
    pd.DataFrame(fold_rows).to_csv(fold_csv_path, index=False, encoding='utf-8')
    print(f"✅ Per-fold summary: {fold_csv_path}")

    # ── Per-sample test results CSV ────────────────────────────────────────
    sample_rows = []
    for r in nested_result['outer_results']:
        for sr in r['llm_test_results']:
            mm = sr['metrics']
            sample_rows.append({
                'outer_fold':       sr['outer_fold'] + 1,
                'sequence':         sr['sequence'],
                'outcome_type':     mm.get('outcome_type', ''),
                'predicted_signs':  ','.join(sr['predicted_signs']),
                'gt_signs':         ','.join(sr['gt_signs']),
                'tp': mm['tp'], 'fp': mm['fp'], 'fn': mm['fn'],
                'precision':        round(mm['precision'], 4),
                'recall':           round(mm['recall'],    4),
                'f1':               round(mm['f1'],        4),
                'null_correct':     mm.get('null_correct', False),
            })
    sample_csv_path = f"{output_dir}/nested_cv_test_samples.csv"
    pd.DataFrame(sample_rows).to_csv(sample_csv_path, index=False, encoding='utf-8')
    print(f"✅ Per-sample test results: {sample_csv_path}")

    # ── Inner CV records CSV ───────────────────────────────────────────────
    inner_rows = []
    for r in nested_result['outer_results']:
        for rec in r['inner']['inner_records']:
            inner_rows.append({
                'outer_fold': r['outer_fold'] + 1,
                'label':      rec['label'],
                'n_train':    rec['n_train'],
                'n_val':      rec['n_val'],
                'opt_f1':     round(rec['opt_f1'],   4),
                'val_f1':     round(rec['val_f1'],   4),
                'baseline':   round(rec['baseline'], 4),
            })
    inner_csv_path = f"{output_dir}/nested_cv_inner_records.csv"
    pd.DataFrame(inner_rows).to_csv(inner_csv_path, index=False, encoding='utf-8')
    print(f"✅ Inner CV records: {inner_csv_path}")

    # ── Global summary JSON ────────────────────────────────────────────────
    summary = {
        'version':        'v1.39_nested_cv',
        'n_outer_folds':  nested_result['n_outer_folds'],
        'outer_n_splits': OUTER_N_SPLITS,
        'inner_n_reps':   INNER_N_REPS,
        'aggregate': {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                          for kk, vv in v.items() if kk != 'values'}
                      for k, v in agg.items()},
    }
    json_path = f"{output_dir}/nested_cv_global_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as _f:
        _json.dump(summary, _f, indent=2, ensure_ascii=False)
    print(f"✅ Global metrics JSON: {json_path}")

    # ── Visualization: box plot of F1 across folds ─────────────────────────
    try:
        llm_vals  = [r['llm_test_metrics']['macro']['f1']  for r in nested_result['outer_results']]
        fast_vals = [r['fast_test_metrics']['macro']['f1'] for r in nested_result['outer_results']]
        base_vals = [r['baseline_metrics']['macro']['f1']  for r in nested_result['outer_results']]
        val_vals  = [r['inner']['val_f1_mean']             for r in nested_result['outer_results']]
        folds     = [f"Fold {r['outer_fold']+1}"          for r in nested_result['outer_results']]

        x = range(OUTER_N_SPLITS)
        w = 0.2
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar([xi-1.5*w for xi in x], val_vals,  w, label='Val 5×2 (rule)', color='steelblue',  alpha=0.8)
        ax.bar([xi-0.5*w for xi in x], fast_vals, w, label='Test fast (opt params)', color='seagreen', alpha=0.8)
        ax.bar([xi+0.5*w for xi in x], llm_vals,  w, label='Test LLM (opt params)',  color='darkorange', alpha=0.8)
        ax.bar([xi+1.5*w for xi in x], base_vals, w, label='Test baseline (v1.28)', color='grey', alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(folds)
        ax.set_ylabel('Macro-F1 (corrected)', fontsize=11)
        ax.set_title('Macro-F1 por Fold Externo — Nested CV v1.32', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        chart_path = f"{output_dir}/nested_cv_f1_per_fold.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Bar chart per fold: {chart_path}")
    except Exception as _e:
        print(f"⚠️  Gráfico não gerado: {_e}")

    # =========================================================================
    # JOURNAL FIGURES (v1.32)
    # =========================================================================
    _plot_journal_figures(nested_result, output_dir)



# ============================================================================
# V1.36 — CONSENSUS CV FUNCTIONS (Strategy B)
# ============================================================================

def _compute_consensus_params(outer_results: List[Dict]) -> List[Dict[str, float]]:
    """
    For each outer fold i, compute consensus params = weighted average of
    the OTHER 4 folds optimised parameters. Weight = best_val_f1.
    Zero data leakage: fold i test sequences were never in any fold j!=i training pool.

    V1.38 FIX: Parameter clipping applied after averaging.
    Clips each parameter to within ±30% of its median across all folds.
    This prevents extreme outlier parameters from one fold dominating the
    consensus for another fold (e.g. Fold 5 catastrophic failure in v1.36-37).
    """
    import numpy as np
    n = len(outer_results)

    # Collect all parameter sets for computing medians
    all_params = [r["inner"]["best_params"] for r in outer_results]
    all_keys   = set()
    for p in all_params: all_keys |= set(p.keys())

    # Compute per-key medians across all 5 folds
    param_medians = {}
    for key in all_keys:
        vals = [p.get(key, 0.0) for p in all_params]
        param_medians[key] = float(np.median(vals))

    consensus_list = []
    for i in range(n):
        other   = [r for j, r in enumerate(outer_results) if j != i]
        weights = [r["inner"]["best_val_f1"] for r in other]
        total_w = sum(weights) or float(len(other))

        # Weighted average
        raw_consensus = {
            key: sum(r["inner"]["best_params"].get(key, 0.0) * w
                     for r, w in zip(other, weights)) / total_w
            for key in all_keys
        }

        # V1.38: clip to ±30% of median to prevent extreme outliers
        clipped = {}
        for key, val in raw_consensus.items():
            med = param_medians[key]
            if med != 0:
                lo  = med * 0.70
                hi  = med * 1.30
                clipped[key] = max(lo, min(hi, val))
            else:
                clipped[key] = val

        consensus_list.append(clipped)
        print(f"  Consensus fold {i+1}: avg of folds "
              f"{[j+1 for j in range(n) if j != i]} "
              f"(weights: {[round(w, 3) for w in weights]}, clipped)")

    return consensus_list


def _run_consensus_cv(df_loo, gt_dict, nested_result, outer_splits):
    """
    Strategy B: evaluate each fold test set with leave-one-fold-out consensus params.
    All test samples in fold i receive identical consensus guidance.
    """
    print("\n" + "="*70)
    print("V1.36 STRATEGY B — CONSENSUS PARAMETER EVALUATION")
    print("="*70)
    print("  Uses v1.35 agent-specific guidance (strict rules — best base).")
    print("  Parameters: leave-one-fold-out weighted average.")

    consensus_params_per_fold = _compute_consensus_params(nested_result["outer_results"])
    consensus_outer = []

    for outer_i, ((tv_idx, test_idx), cons_p) in enumerate(
            zip(outer_splits, consensus_params_per_fold)):

        print(f"\n{chr(8211)*70}")
        print(f"CONSENSUS FOLD {outer_i+1}/5  (test={len(test_idx)} sequences)")
        print(f"{chr(8211)*70}")

        train_val_df = df_loo.iloc[tv_idx].reset_index(drop=True)
        test_df      = df_loo.iloc[test_idx].reset_index(drop=True)

        apply_optimized_params(cons_p)
        _apply_params_to_score_candidate()

        # Fast test with consensus params
        test_samples = _precompute_loo_samples(test_df, gt_dict)
        fast_res = []
        for s in test_samples:
            pred = _select_dets_parametric(s, cons_p)
            m    = evaluate_single_prediction(pred, s["gt_signs"])
            fast_res.append({"loo_index": 0, "outer_fold": outer_i,
                             "sequence": s["sequence"],
                             "predicted_signs": pred, "gt_signs": s["gt_signs"],
                             "metrics": m, "few_shot_n": 0})
        fast_metrics = compute_loo_metrics(fast_res)

        # LLM test with consensus guidance
        print(f"  Evaluating {len(test_df)} samples with consensus guidance...")
        llm_res = _evaluate_test_with_llm(
            test_df, train_val_df, gt_dict, cons_p, TEST_N_FEW_SHOT, outer_i)
        llm_metrics = compute_loo_metrics(llm_res)

        # Comparison with Strategy A
        fold_a = nested_result["outer_results"][outer_i]
        sm     = fold_a["llm_test_metrics"]
        print(f"\n  FOLD {outer_i+1} COMPARISON (A=fold-specific, B=consensus):")
        print(f"  {'Metric':<30} {'A':>10} {'B':>10} {'Delta':>10}")
        print(f"  {'-'*54}")
        for label, a_v, b_v in [
            ("LLM Macro-F1 corrected", sm["macro"]["f1"],  llm_metrics["macro"]["f1"]),
            ("LLM Micro-F1",           sm["micro"]["f1"],  llm_metrics["micro"]["f1"]),
            ("Fast Macro-F1",          fold_a["fast_test_metrics"]["macro"]["f1"],
                                       fast_metrics["macro"]["f1"]),
        ]:
            d = b_v - a_v
            sign = '+ ' if d >= 0 else ''
            print(f"  {label:<30} {a_v*100:>9.2f}% {b_v*100:>9.2f}% "
                  f"{sign}{d*100:>8.2f}pp")

        consensus_outer.append({
            "outer_fold":        outer_i, "n_test": len(test_idx),
            "consensus_params":  cons_p,
            "llm_test_metrics":  llm_metrics,
            "fast_test_metrics": fast_metrics,
            "llm_test_results":  llm_res,
        })

    return _aggregate_consensus_results(consensus_outer, nested_result)


def _aggregate_consensus_results(consensus_outer, nested_result):
    """Aggregates consensus CV results."""
    import numpy as np
    def _c(key_path):
        vals = []
        for r in consensus_outer:
            obj = r
            for k in key_path: obj = obj[k]
            vals.append(float(obj))
        return np.array(vals)
    agg = {
        "llm_macro_f1":  _c(["llm_test_metrics","macro","f1"]),
        "llm_micro_f1":  _c(["llm_test_metrics","micro","f1"]),
        "fast_macro_f1": _c(["fast_test_metrics","macro","f1"]),
    }
    return {
        "consensus_outer_results": consensus_outer,
        "aggregate": {k: {"mean": float(v.mean()), "std": float(v.std()),
                          "values": v.tolist()} for k, v in agg.items()},
        "nested_aggregate": nested_result["aggregate"],
    }


def print_strategy_comparison(consensus_result):
    """Prints side-by-side comparison of Strategy A vs Strategy B."""
    agg_c = consensus_result["aggregate"]
    agg_n = consensus_result["nested_aggregate"]
    def _f(agg, key):
        d = agg.get(key, {})
        if "mean" not in d: return "N/A"
        return f"{d['mean']*100:.2f}% +/-{d['std']*100:.2f}%"
    print("\n" + "="*72)
    print("STRATEGY COMPARISON — v1.36 (v1.35 guidance base)")
    print("="*72)
    print(f"  {'Metric':<35} {'A (fold-specific)':>18} {'B (consensus)':>15}")
    print("  " + "-"*70)
    for label, key_n, key_c in [
        ("LLM Macro-F1 corrected",   "llm_macro_f1",      "llm_macro_f1"),
        ("LLM Micro-F1",             "llm_micro_f1",      "llm_micro_f1"),
        ("Fast Macro-F1 (rules)",    "fast_macro_f1",     "fast_macro_f1"),
        ("Inner Val F1",             "val_f1_mean",       None),
        ("Baseline Macro-F1",        "baseline_macro_f1", None),
    ]:
        a = _f(agg_n, key_n)
        b = _f(agg_c, key_c) if key_c else "n/a"
        print(f"  {label:<35} {a:>18} {b:>15}")
    if "llm_macro_f1" in agg_c and "llm_macro_f1" in agg_n:
        d = agg_c["llm_macro_f1"]["mean"] - agg_n["llm_macro_f1"]["mean"]
        winner = 'consensus wins' if d > 0 else 'fold-specific wins'
        print(f"\n  Delta LLM Macro-F1 (B-A): {d*100:+.2f}pp  ({winner})")
    print(f"\n  {'Fold':<6} {'A: LLM F1':>12} {'B: LLM F1':>12} {'Delta':>10}")
    print("  " + "-"*42)
    vals_a = agg_n.get("llm_macro_f1", {}).get("values", [])
    vals_b = agg_c.get("llm_macro_f1", {}).get("values", [])
    for i, (a, b) in enumerate(zip(vals_a, vals_b)):
        d = b - a
        print(f"  {i+1:<6} {a*100:>11.2f}% {b*100:>11.2f}% {d*100:>+9.2f}pp")
    print("="*72 + "\n")


def _plot_strategy_comparison(consensus_result, output_dir):
    """Figure comparing Strategy A vs B per fold and overall distribution."""
    import numpy as np
    agg_c  = consensus_result["aggregate"]
    agg_n  = consensus_result["nested_aggregate"]
    vals_a = agg_n.get("llm_macro_f1", {}).get("values", [])
    vals_b = agg_c.get("llm_macro_f1", {}).get("values", [])
    if not vals_a or not vals_b:
        print("  No data for strategy comparison figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = len(vals_a); x = np.arange(n); w = 0.35
    ax = axes[0]
    ax.bar(x-w/2, [v*100 for v in vals_a], w,
           label="Strategy A (fold-specific)", color="#2166AC", alpha=0.85)
    ax.bar(x+w/2, [v*100 for v in vals_b], w,
           label="Strategy B (consensus)", color="#D6604D", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"Fold {i+1}" for i in range(n)])
    ax.set_ylabel("LLM Macro-F1 (null-corrected)"); ax.set_ylim(0, 105)
    ax.axhline(np.mean(vals_a)*100, color="#2166AC", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(np.mean(vals_b)*100, color="#D6604D", linestyle="--", linewidth=1, alpha=0.6)
    ax.legend(fontsize=9)
    ax.set_title("Per-Fold LLM Macro-F1\nStrategy A vs Strategy B")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    ax2 = axes[1]
    bp = ax2.boxplot([[v*100 for v in vals_a], [v*100 for v in vals_b]],
                     patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], ["#2166AC", "#D6604D"]):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    for i, vals in enumerate([[v*100 for v in vals_a], [v*100 for v in vals_b]], 1):
        ax2.scatter([i]*len(vals), vals, color="black", s=40, zorder=5, alpha=0.8)
    ax2.set_xticklabels(["Strategy A\n(fold-specific)", "Strategy B\n(consensus)"])
    ax2.set_ylabel("LLM Macro-F1 (%)"); ax2.set_title("Distribution Across Folds")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    plt.suptitle("v1.36: Strategy A (v1.35 guidance) vs Strategy B (consensus)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        kw = {"bbox_inches":"tight"} if ext==".pdf" else {"dpi":300,"bbox_inches":"tight"}
        plt.savefig(f"{output_dir}/fig_strategy_comparison{ext}", **kw)
    plt.close()
    print(f"  Strategy comparison figure saved.")


def _export_consensus_results(consensus_result, output_dir):
    """Export CSV and JSON for consensus CV results."""
    import json as _json
    rows = []
    for r in consensus_result["consensus_outer_results"]:
        lm = r["llm_test_metrics"]; fm = r["fast_test_metrics"]
        ns = lm["null_stats"]
        rows.append({"outer_fold": r["outer_fold"]+1, "strategy": "B_consensus",
                     "n_test": r["n_test"],
                     "llm_macro_f1": round(lm["macro"]["f1"], 4),
                     "llm_micro_f1": round(lm["micro"]["f1"], 4),
                     "fast_macro_f1": round(fm["macro"]["f1"], 4),
                     "null_accuracy": round(ns["null_accuracy"], 4)
                                      if ns.get("null_accuracy") else ""})
    pd.DataFrame(rows).to_csv(f"{output_dir}/consensus_cv_per_fold.csv", index=False)
    agg = consensus_result["aggregate"]
    _json.dump({"version": "v1.36_consensus_cv",
                "strategy": "B_leave_one_fold_out_consensus",
                "guidance_base": "v1.35_strict_agent_specific",
                "aggregate": {k: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)}
                              for k, v in agg.items()}},
               open(f"{output_dir}/consensus_cv_global_metrics.json", "w"), indent=2)
    print("  Consensus CV results exported.")


def _plot_journal_figures(nested_result: Dict, output_dir: str) -> None:
    """
    Generates all six journal-quality figures for the nested CV evaluation.

    Figure 1 — Box plot: macro-F1 distribution across folds (3 conditions)
    Figure 2 — Scatter: validation vs test F1 (generalisation gap)
    Figure 3 — Horizontal bar: per-sign F1 (colour-coded by category)
    Figure 4 — Stacked bar: outcome type distribution per fold
    Figure 5 — Line: optimisation convergence (inner CV, supplementary)
    Figure 6 — Grouped bar: precision and recall per determinative category
    """
    import numpy as np
    from collections import defaultdict

    # ── Shared style ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':      'serif',
        'font.size':        11,
        'axes.titlesize':   13,
        'axes.labelsize':   12,
        'xtick.labelsize':  10,
        'ytick.labelsize':  10,
        'legend.fontsize':  10,
        'figure.dpi':       150,
    })

    COLORS = {
        'llm':      '#2166AC',   # dark blue
        'fast':     '#4DAC26',   # green
        'baseline': '#888888',   # grey
        'val':      '#D1E5F0',   # light blue
        'TRUE_NULL':'#4DAC26',
        'STANDARD': '#2166AC',
        'FP_NULL':  '#F4A582',
        'FN_NULL':  '#D6604D',
    }

    outer_results = nested_result['outer_results']
    n_folds       = len(outer_results)
    fold_labels   = [f"Fold {r['outer_fold']+1}" for r in outer_results]

    # Collect all test results across folds
    all_test = [sr for r in outer_results for sr in r['llm_test_results']]

    # =========================================================================
    # FIGURE 1 — Box plot: macro-F1 across folds, three conditions
    # =========================================================================
    try:
        llm_vals  = [r['llm_test_metrics']['macro']['f1']  for r in outer_results]
        fast_vals = [r['fast_test_metrics']['macro']['f1'] for r in outer_results]
        base_vals = [r['baseline_metrics']['macro']['f1']  for r in outer_results]
        val_vals  = [r['inner']['val_f1_mean']             for r in outer_results]

        fig, ax = plt.subplots(figsize=(8, 5))
        data      = [base_vals, fast_vals, val_vals, llm_vals]
        labels    = ['Baseline\n(v1.28 rules)',
                     'Optimised\nrules',
                     'Inner CV val\n(5×2, rules)',
                     'LLM +\nopt. params']
        colors    = [COLORS['baseline'], COLORS['fast'], COLORS['val'], COLORS['llm']]

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        # Overlay individual fold points
        for i, vals in enumerate(data, 1):
            ax.scatter([i]*len(vals), vals, color='black',
                       s=40, zorder=5, alpha=0.7)

        ax.set_xticklabels(labels)
        ax.set_ylabel('Macro-F1 (null-corrected)')
        ax.set_title('Figure 1 — Performance Distribution Across Folds\n'
                     '(5 outer folds, 19–20 test sequences each)')
        ax.set_ylim(0, 1.05)
        ax.axhline(np.mean(llm_vals), color=COLORS['llm'],
                   linestyle='--', linewidth=1, alpha=0.6,
                   label=f'LLM mean = {np.mean(llm_vals)*100:.1f}%')
        ax.legend(loc='lower right')
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
        plt.tight_layout()
        p = f"{output_dir}/fig1_boxplot_f1.pdf"
        plt.savefig(p, bbox_inches='tight'); plt.savefig(p.replace('.pdf','.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Figure 1: {p}")
    except Exception as _e:
        print(f"⚠️  Figure 1 failed: {_e}")

    # =========================================================================
    # FIGURE 2 — Scatter: validation F1 vs test F1 (generalisation gap)
    # =========================================================================
    try:
        val_means = [r['inner']['val_f1_mean']            for r in outer_results]
        val_stds  = [r['inner']['val_f1_std']             for r in outer_results]
        test_f1s  = [r['llm_test_metrics']['macro']['f1'] for r in outer_results]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4, label='y = x (ideal)')

        for i, (vf, vs, tf, fl) in enumerate(
                zip(val_means, val_stds, test_f1s, fold_labels)):
            ax.errorbar(vf, tf, xerr=vs, fmt='o',
                        color=COLORS['llm'], markersize=9,
                        capsize=4, elinewidth=1.2, ecolor='steelblue',
                        zorder=5)
            ax.annotate(fl, (vf, tf),
                        textcoords='offset points', xytext=(8, 4),
                        fontsize=9)

        gap = np.mean(test_f1s) - np.mean(val_means)
        ax.set_xlabel('Validation Macro-F1  (inner 5×2 CV, rules)')
        ax.set_ylabel('Test Macro-F1  (LLM + optimised params)')
        ax.set_title('Figure 2 — Generalisation: Validation vs Test F1\n'
                     f'Mean gap = {gap*100:+.1f}pp  '
                     f'({"overfitting" if gap < 0 else "generalises well"})')
        lo = min(min(val_means), min(test_f1s)) - 0.05
        hi = max(max(val_means), max(test_f1s)) + 0.05
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f'{y*100:.0f}%'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f'{y*100:.0f}%'))
        ax.legend()
        plt.tight_layout()
        p = f"{output_dir}/fig2_scatter_val_vs_test.pdf"
        plt.savefig(p, bbox_inches='tight'); plt.savefig(p.replace('.pdf','.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Figure 2: {p}")
    except Exception as _e:
        print(f"⚠️  Figure 2 failed: {_e}")

    # =========================================================================
    # FIGURE 3 — Horizontal bar: per-sign F1 (min 3 GT occurrences)
    # =========================================================================
    try:
        from collections import Counter

        # Aggregate TP/FP/FN per sign across all folds
        sign_tp = defaultdict(int); sign_fp = defaultdict(int)
        sign_fn = defaultdict(int); sign_gt = defaultdict(int)

        for sr in all_test:
            from collections import Counter as _C
            pm = dict(_C(sr['predicted_signs']))
            gm = dict(_C(sr['gt_signs']))
            for s in set(pm) | set(gm):
                sign_gt[s] += gm.get(s, 0)
                sign_tp[s] += min(pm.get(s,0), gm.get(s,0))
                sign_fp[s] += max(0, pm.get(s,0) - gm.get(s,0))
                sign_fn[s] += max(0, gm.get(s,0) - pm.get(s,0))

        # Filter: at least 3 GT occurrences
        eligible = {s for s, n in sign_gt.items() if n >= 3}
        sign_f1  = {}
        for s in eligible:
            tp, fp, fn = sign_tp[s], sign_fp[s], sign_fn[s]
            p_ = tp/(tp+fp) if tp+fp else 0
            r_ = tp/(tp+fn) if tp+fn else 0
            f_ = 2*p_*r_/(p_+r_) if p_+r_ else 0
            sign_f1[s] = (f_, sign_gt[s])

        if sign_f1:
            sorted_signs = sorted(sign_f1.items(), key=lambda x: x[1][0])
            signs   = [s for s, _ in sorted_signs]
            f1s     = [v[0] for _, v in sorted_signs]
            gt_cnts = [v[1] for _, v in sorted_signs]

            # Category colour map
            # V1.33: CAT_COLORS aligned with actual category names in DETERMINATIVES_DETAILED
            CAT_COLORS = {
                'person':    '#2166AC',   # dark blue
                'divine':    '#762A83',   # purple
                'animal':    '#4DAC26',   # green
                'abstract':  '#F4A582',   # salmon
                'land':      '#D6604D',   # red-orange
                'grammar':   '#FDAE61',   # orange
                'object':    '#878787',   # grey
                'nature':    '#1B7837',   # dark green
                'body':      '#80CDC1',   # teal
                'action':    '#DFC27D',   # tan
                'sky':       '#92C5DE',   # light blue
                'building':  '#A6611A',   # brown
                'transport': '#E7D4E8',   # light purple
                'food':      '#FDB863',   # light orange
                'funeral':   '#543005',   # dark brown
                'tool':      '#BF812D',   # gold
                'material':  '#C7EAE5',   # light teal
                'unknown':   '#CCCCCC',   # light grey
            }
            bar_colors = []
            for s in signs:
                cat = DETERMINATIVES_DETAILED.get(s, {}).get('categoria', 'unknown')
                bar_colors.append(CAT_COLORS.get(cat, '#AAAAAA'))

            fig_h = max(5, len(signs) * 0.38)
            fig, ax = plt.subplots(figsize=(9, fig_h))
            y_pos = range(len(signs))
            bars  = ax.barh(y_pos, f1s, color=bar_colors, alpha=0.85, height=0.7)

            # GT count annotation
            for bar, cnt in zip(bars, gt_cnts):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'n={cnt}', va='center', fontsize=8, color='#444444')

            ax.set_yticks(y_pos); ax.set_yticklabels(signs, fontsize=9)
            ax.set_xlabel('F1-Score')
            ax.set_xlim(0, 1.12)
            ax.axvline(0.5, color='grey', linestyle=':', linewidth=1)
            ax.set_title('Figure 3 — Per-Sign F1 Score\n'
                         '(signs with ≥3 GT occurrences; colour = category)')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f'{y*100:.0f}%'))

            # Legend for categories
            from matplotlib.patches import Patch
            # Legend: only categories that actually appear in the filtered signs
            cats_present = {DETERMINATIVES_DETAILED.get(s, {}).get('categoria', 'unknown')
                            for s in signs}
            legend_elements = [Patch(facecolor=CAT_COLORS[cat], label=cat, alpha=0.85)
                               for cat in sorted(CAT_COLORS.keys())
                               if cat in cats_present]
            if legend_elements:
                ax.legend(handles=legend_elements, loc='lower right',
                          fontsize=8, ncol=2)
            plt.tight_layout()
            p = f"{output_dir}/fig3_per_sign_f1.pdf"
            plt.savefig(p, bbox_inches='tight'); plt.savefig(p.replace('.pdf','.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Figure 3: {p}")
        else:
            print("⚠️  Figure 3: no signs with ≥3 GT occurrences")
    except Exception as _e:
        print(f"⚠️  Figure 3 failed: {_e}")

    # =========================================================================
    # FIGURE 4 — Stacked bar: outcome type distribution per fold
    # =========================================================================
    try:
        outcome_types = ['TRUE_NULL', 'STANDARD', 'FP_NULL', 'FN_NULL']
        outcome_counts = {ot: [] for ot in outcome_types}

        for r in outer_results:
            ns = r['llm_test_metrics']['null_stats']
            outcome_counts['TRUE_NULL'].append(ns.get('n_true_null', 0))
            outcome_counts['FP_NULL'].append(ns.get('n_fp_null',   0))
            outcome_counts['FN_NULL'].append(ns.get('n_fn_null',   0))
            outcome_counts['STANDARD'].append(ns.get('n_standard', 0))

        fig, ax = plt.subplots(figsize=(9, 5))
        x       = np.arange(n_folds)
        bottom  = np.zeros(n_folds)
        labels_map = {
            'TRUE_NULL': 'True Null (pred=∅, GT=∅)',
            'STANDARD':  'Standard (pred≠∅, GT≠∅)',
            'FP_NULL':   'FP Null (pred≠∅, GT=∅)',
            'FN_NULL':   'FN Null (pred=∅, GT≠∅)',
        }
        for ot in outcome_types:
            vals = np.array(outcome_counts[ot])
            ax.bar(x, vals, bottom=bottom,
                   label=labels_map[ot],
                   color=COLORS[ot], alpha=0.85, width=0.55)
            # Value labels inside bars
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0:
                    ax.text(xi, b + v/2, str(int(v)),
                            ha='center', va='center',
                            fontsize=9, fontweight='bold', color='white')
            bottom += vals

        ax.set_xticks(x); ax.set_xticklabels(fold_labels)
        ax.set_ylabel('Number of test sequences')
        ax.set_title('Figure 4 — Prediction Outcome Distribution per Fold')
        ax.legend(loc='upper right', fontsize=9)
        plt.tight_layout()
        p = f"{output_dir}/fig4_outcome_distribution.pdf"
        plt.savefig(p, bbox_inches='tight'); plt.savefig(p.replace('.pdf','.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Figure 4: {p}")
    except Exception as _e:
        print(f"⚠️  Figure 4 failed: {_e}")

    # =========================================================================
    # FIGURE 5 — Line: optimisation convergence (supplementary)
    # =========================================================================
    try:
        fig, axes = plt.subplots(1, n_folds, figsize=(4*n_folds, 4),
                                 sharey=True)
        if n_folds == 1:
            axes = [axes]

        for ax, r in zip(axes, outer_results):
            records   = r['inner']['inner_records']
            rep_labels= [rec['label'] for rec in records]
            val_f1s   = [rec['val_f1']   for rec in records]
            opt_f1s   = [rec['opt_f1']   for rec in records]
            init_f1s  = [rec.get('init_f1', rec['opt_f1'])  for rec in records]
            baselines = [rec['baseline'] for rec in records]
            xs = range(1, len(records)+1)

            ax.plot(xs, val_f1s,   'o-', color=COLORS['llm'],
                    linewidth=2, markersize=6, label='Val F1')
            ax.plot(xs, opt_f1s,   's--', color=COLORS['fast'],
                    linewidth=1.5, markersize=5, label='Opt F1 (train fold)')
            ax.plot(xs, init_f1s,  'D:', color='purple',
                    linewidth=1.5, markersize=4, label=f'Init F1 ({INIT_STRATEGY})')
            ax.plot(xs, baselines, '^:', color=COLORS['baseline'],
                    linewidth=1, markersize=4, label='Baseline v1.28')
            ax.set_xticks(xs)
            ax.set_xticklabels([r.split('_')[0] for r in rep_labels],
                               rotation=30, fontsize=8)
            ax.set_title(f"Fold {r['outer_fold']+1}", fontsize=11)
            ax.set_xlabel('Inner rep')
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y,_: f'{y*100:.0f}%'))

        axes[0].set_ylabel('F1-Score')
        axes[0].legend(loc='lower left', fontsize=8)
        fig.suptitle('Figure 5 (Supplementary) — Inner 5×2 CV Convergence\n'
                     'per Outer Fold', fontsize=12, fontweight='bold')
        plt.tight_layout()
        p = f"{output_dir}/fig5_supp_convergence.pdf"
        plt.savefig(p, bbox_inches='tight'); plt.savefig(p.replace('.pdf','.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Figure 5 (supplementary): {p}")
    except Exception as _e:
        print(f"⚠️  Figure 5 failed: {_e}")

    # =========================================================================
    # FIGURE 6 — Grouped bar: precision and recall per category
    # =========================================================================
    try:
        cat_tp = defaultdict(int); cat_fp = defaultdict(int)
        cat_fn = defaultdict(int)

        for sr in all_test:
            from collections import Counter as _C
            pm = dict(_C(sr['predicted_signs']))
            gm = dict(_C(sr['gt_signs']))
            for s in set(pm) | set(gm):
                cat = DETERMINATIVES_DETAILED.get(s, {}).get('categoria', 'Unknown')
                tp_ = min(pm.get(s,0), gm.get(s,0))
                fp_ = max(0, pm.get(s,0) - gm.get(s,0))
                fn_ = max(0, gm.get(s,0) - pm.get(s,0))
                cat_tp[cat] += tp_; cat_fp[cat] += fp_; cat_fn[cat] += fn_

        # Compute P, R, F1 per category
        cat_metrics = {}
        for cat in sorted(cat_tp.keys()):
            tp_, fp_, fn_ = cat_tp[cat], cat_fp[cat], cat_fn[cat]
            p_ = tp_/(tp_+fp_) if tp_+fp_ else 0
            r_ = tp_/(tp_+fn_) if tp_+fn_ else 0
            f_ = 2*p_*r_/(p_+r_) if p_+r_ else 0
            cat_metrics[cat] = (p_, r_, f_)

        if cat_metrics:
            cats  = list(cat_metrics.keys())
            precs = [cat_metrics[c][0] for c in cats]
            recs  = [cat_metrics[c][1] for c in cats]
            f1s_c = [cat_metrics[c][2] for c in cats]

            x  = np.arange(len(cats))
            w  = 0.27
            fig, ax = plt.subplots(figsize=(max(8, len(cats)*1.1), 5))
            ax.bar(x - w,   precs, w, label='Precision',
                   color='#2166AC', alpha=0.85)
            ax.bar(x,       recs,  w, label='Recall',
                   color='#4DAC26', alpha=0.85)
            ax.bar(x + w,   f1s_c, w, label='F1',
                   color='#D6604D', alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(cats, rotation=30, ha='right')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
            ax.axhline(0.5, color='grey', linestyle=':', linewidth=1)
            ax.set_title('Figure 6 — Precision, Recall and F1 per Determinative Category')
            ax.legend()
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y,_: f'{y*100:.0f}%'))
            plt.tight_layout()
            p = f"{output_dir}/fig6_category_prf.pdf"
            plt.savefig(p, bbox_inches='tight'); plt.savefig(p.replace('.pdf','.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Figure 6: {p}")
        else:
            print("⚠️  Figure 6: no category data available")
    except Exception as _e:
        print(f"⚠️  Figure 6 failed: {_e}")

DATA_PATH = "F:/estancia_espanha_agentes/content/extractrion_for_agents.xlsx"
OUTPUT_DIR = "F:/estancia_espanha_agentes/content/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# ⚙️  CONFIGURAÇÕES LOO (v1.29)
# ============================================================================

# Caminhos para os arquivos DOCX com anotações do especialista (gabarito LOO)
# Altere para os caminhos corretos no seu sistema:
DOCX_PATHS = [
    "F:/estancia_espanha_agentes/content/Langgraph_analyses_comments.docx",
    "F:/estancia_espanha_agentes/content/Langgraph_analyses_comments_2.docx",
]

# ─── Modo de execução LOO ────────────────────────────────────────────────────
# True  → executa avaliação LOO completa (99 iterações, ~99x mais lento)
# False → executa análise normal (comportamento original v1.28)
LOO_MODE = True

# Número de exemplos few-shot por iteração LOO (recomendado: 5)
LOO_N_FEW_SHOT = 5

# Para debugging: limite de amostras LOO a processar (None = todas as 99)
LOO_MAX_SAMPLES = None  # ex: 10 para testar apenas as 10 primeiras

# ─── Otimização de hiperparâmetros (v1.30) ──────────────────────────────────
# True  → roda otimização Bayesiana (Optuna) ANTES do LOO principal
#         Os melhores parâmetros encontrados substituem os valores hardcoded
#         e são usados no LOO final de avaliação.
# False → usa os parâmetros hardcoded originais (comportamento v1.29)
OPTIMIZE_PARAMS = True

# Número de tentativas do Optuna (mais tentativas = melhor busca, mais lento)
# Recomendado: 200-500 para busca séria; 50 para teste rápido.
N_OPTUNA_TRIALS = 200

# Salvar/carregar parâmetros já otimizados (evita re-otimizar toda vez)
# None → não salva/carrega; string → caminho do arquivo JSON
PARAMS_CACHE_PATH = f"{OUTPUT_DIR}/optimized_params_v1_30.json" \
    if 'OUTPUT_DIR' in dir() else None

# ─── Nested CV (v1.32) ───────────────────────────────────────────────────────
# Modo de avaliação principal. Escolha UMA das opções:
#   'nested_cv' → nova estratégia 5×(5×2 CV) com teste separado  ← RECOMENDADO
#   'loo'       → Leave-One-Out completo (v1.29/v1.30/v1.31)
#   'normal'    → processamento normal sem avaliação (v1.28)
EVAL_MODE = 'nested_cv'

# ── Outer loop (5 splits 80/19) ──────────────────────────────────────────────
# Número de vezes que o dataset é dividido em 80 (treino/val) + 19 (teste)
OUTER_N_SPLITS = 5          # 5 splits → 5 test sets diferentes (~19 amostras cada)
OUTER_RANDOM_STATE = GLOBAL_SEED  # v1.42: centralised     # seed para reprodutibilidade do shuffle externo

# ── Inner loop (5×2 cross-validation nos 80) ─────────────────────────────────
# Para cada split externo, os 80 são divididos 50/50 (40+40) e o processo
# é repetido INNER_N_REPS vezes com shuffles diferentes.
# Cada repetição produz 2 estimativas de validação (treino A→val B, treino B→val A).
# Total de avaliações de validação por split externo: INNER_N_REPS × 2 = 10
INNER_N_REPS    = 5         # repetições do 2-fold interno (Dietterich 5×2 CV)
INNER_N_TRIALS  = 400       # trials Optuna por half-fold (aumentado para B+D — mais exploração)

# ── Estratégia de inicialização (Options B + D — v1.33) ──────────────────────
# 'default'  → warm-start nos valores v1.28 (comportamento original)
# 'flat'     → Option B: pesos iguais, sem bonificações, sem penalidades (prior plano)
# 'random'   → ponto de partida aleatório (exploração máxima, mais variância)
INIT_STRATEGY = 'flat'   # ← Option B: prior plano uninformativo

# ── Pesos de fontes LLM (Option D — separados dos pesos rule-only) ────────────
# Quando USE_LANGGRAPH=True, o scoring usa w_sem_llm / w_struct_llm / w_rule_llm
# em vez de w_sem / w_struct / w_rule, permitindo calibração independente.
# Inicializados pelo prior plano (Option B): todos iguais = 0.33.
# O inner CV não consegue optimizá-los (sem LLM), mas podem ser ajustados
# manualmente ou por uma fase adicional de optimização LLM-aware.
LLM_SOURCE_WEIGHTS_INIT = {
    'w_sem_llm':    0.0,
    'w_struct_llm': 0.0,
    'w_rule_llm':   1.0,
}

# ── Teste com LLM ────────────────────────────────────────────────────────────
TEST_N_FEW_SHOT = 5

# ── Strategy B: Consensus CV (v1.36) ─────────────────────────────────────────
# When True, runs a second evaluation after nested CV using consensus params.
# For each outer fold i: consensus = weighted avg of other 4 folds' params.
# Zero data leakage: fold i test sequences were never in any fold j≠i training pool.
# Allows direct comparison of:
#   Strategy A: fold-specific params + guidance  (best mean: 80.1% from v1.35)
#   Strategy B: consensus params + guidance      (stable, lower variance)
RUN_CONSENSUS_CV = True

# Verificar dependências do LangGraph
def verificar_dependencias_langgraph():
    """Verifica se as dependências do LangGraph estão instaladas"""
    dependencias = {
        'langgraph': 'langgraph',
        'langchain_openai': 'langchain-openai',
        'langchain_anthropic': 'langchain-anthropic',
        'langchain_core': 'langchain-core'
    }
    
    faltando = []
    for modulo, pacote in dependencias.items():
        try:
            __import__(modulo)
        except ImportError:
            faltando.append(pacote)
    
    if faltando:
        print(f"⚠️  Dependências faltando para LangGraph: {', '.join(faltando)}")
        print(f"   Instale com: pip install {' '.join(faltando)}")
        return False
    
    return True

# Verificar se há API key disponível para LangGraph
USE_LANGGRAPH = False
has_openai = 'OPENAI_API_KEY' in os.environ
has_anthropic = 'ANTHROPIC_API_KEY' in os.environ

if has_openai or has_anthropic:
    if verificar_dependencias_langgraph():
        USE_LANGGRAPH = True
        print("🤖 API key detectada e dependências OK - ¡LangGraph se utilizará!")
        if has_openai:
            print("   Provider: OpenAI")
        elif has_anthropic:
            print("   Provider: Anthropic")
    else:
        print("⚠️  API key detectada mas dependências faltando")
        print("   LangGraph será desativado")
else:
    print("⚠️  No se detectó ninguna API key - usando análisis basado en reglas")
    print("   Para usar LangGraph, configure OPENAI_API_KEY o ANTHROPIC_API_KEY")

#print(f"🖥️  Dispositivo: {DEVICE}")
print(f"📁 Directorio de salida: {OUTPUT_DIR}")
print(f"🤖 Modo LangGraph: {'ACTIVO' if USE_LANGGRAPH else 'INACTIVO (solo reglas)'}\n")

#%%
# ============================================================================
# FASE 2 - CARREGAMENTO E PARSING DOS DADOS
# ============================================================================

print("📂 Cargando datos...")

df = pd.read_excel(DATA_PATH)

# ============================================================================
# Inicializar Few-Shot Learning com exemplos do especialista
# ============================================================================
try:
    EXPERT_EXAMPLES = select_expert_examples(df, n_examples=5)
    print(f"✅ Few-Shot ativado: {len(EXPERT_EXAMPLES)} exemplos carregados")
except Exception as e:
    print(f"⚠️  Few-Shot desabilitado: {e}")
    EXPERT_EXAMPLES = []
# ============================================================================

print(f"✅ CSV cargado: {len(df)} linhas")

# ============================================================================
# V1.29: CARREGAR GABARITO LOO DOS ARQUIVOS DOCX
# ============================================================================
LOO_GT_DICT = {}   # normalized_seq → list[str] (bold GT determinatives)
LOO_DF = None       # subset of df matching the 99 annotated sequences

if LOO_MODE:
    print("\n🔬 V1.29 LOO: Carregando gabarito dos arquivos DOCX...")
    # Resolve paths: also try same folder as DATA_PATH if DOCX_PATHS not found
    import os.path as _osp
    _data_dir = _osp.dirname(DATA_PATH)
    _resolved_docx = []
    for _p in DOCX_PATHS:
        if _osp.exists(_p):
            _resolved_docx.append(_p)
        else:
            _alt = _osp.join(_data_dir, _osp.basename(_p))
            if _osp.exists(_alt):
                _resolved_docx.append(_alt)
            else:
                print(f"  ⚠️  DOCX não encontrado: {_p}")
    if _resolved_docx:
        LOO_GT_DICT = load_loo_ground_truth(_resolved_docx)
    else:
        print("  ⚠️  Nenhum DOCX encontrado. LOO_MODE será desativado.")
        LOO_MODE = False

    if LOO_GT_DICT:
        # Build df subset: only rows whose normalized code matches a GT key
        _gt_keys = set(LOO_GT_DICT.keys())
        _mask = df['Code'].apply(lambda c: re.sub(r'\s+', '', str(c).strip()) if isinstance(c, str) else '').isin(_gt_keys)
        LOO_DF = df[_mask].copy().reset_index(drop=True)
        print(f"✅ df_loo construído: {len(LOO_DF)} linhas com gabarito")
        if len(LOO_DF) != 99:
            print(f"  ⚠️  Esperado 99, encontrado {len(LOO_DF)}. Verifique normalização das sequências.")
    else:
        LOO_MODE = False
        print("  ⚠️  GT dict vazio — LOO_MODE desativado.")

# ── Inicializar parâmetros ativos (default; serão substituídos se otimizado) ─
_ACTIVE_PARAMS: Dict[str, float] = _default_params()

# ============================================================================
# V1.30: OTIMIZAÇÃO DE HIPERPARÂMETROS (Stage 1 — sem LLM)
#
# IMPORTANTE: Em modo 'nested_cv' este bloco NÃO executa, porque a otimização
# global usaria todas as 99 amostras (incluindo as que serão teste em cada fold
# externo) — causando data leakage. O nested CV faz a sua própria otimização
# por fold, restrita aos 80 amostras de treino/validação.
# ============================================================================

if (OPTIMIZE_PARAMS and EVAL_MODE == 'loo'
        and LOO_GT_DICT and LOO_DF is not None):
    print("\n" + "="*70)
    print("⚙️  V1.30: OTIMIZAÇÃO DE HIPERPARÂMETROS (LOO + Optuna)")
    print("="*70)

    # Pré-computar dados de todas as 99 amostras UMA vez
    print(f"\n📦 Pré-computando {len(LOO_DF)} amostras (tokenização + regras)...")
    _precomputed_samples = _precompute_loo_samples(LOO_DF, LOO_GT_DICT)
    print(f"✅ Pré-computação concluída: {len(_precomputed_samples)} amostras prontas")

    # Rodar otimização Bayesiana
    _best_params = optimize_parameters_loo(
        _precomputed_samples,
        n_trials=N_OPTUNA_TRIALS,
        cache_path=PARAMS_CACHE_PATH,
    )

    # Comparar default vs otimizado
    print_param_comparison(_default_params(), _best_params)

    # Aplicar parâmetros otimizados ao pipeline
    apply_optimized_params(_best_params)
    _apply_params_to_score_candidate()

    # Salvar tabela de melhores parâmetros para referência
    _params_table_path = f"{OUTPUT_DIR}/optimized_params_table.csv"
    _default_for_compare = _default_params()
    _rows = [{'parameter': k,
              'default':   round(_default_for_compare.get(k, float('nan')), 4),
              'optimized': round(_best_params.get(k, float('nan')), 4),
              'delta':     round(_best_params.get(k, 0) - _default_for_compare.get(k, 0), 4)}
             for k in sorted(set(_default_for_compare) | set(_best_params))]
    pd.DataFrame(_rows).to_csv(_params_table_path, index=False, encoding='utf-8')
    print(f"✅ Tabela de parâmetros: {_params_table_path}")

elif EVAL_MODE == 'nested_cv':
    print("\nℹ️  EVAL_MODE='nested_cv': otimização global desativada (evita data leakage).")
    print("   Otimização será feita por fold externo dentro do nested CV.")

elif OPTIMIZE_PARAMS and (not LOO_GT_DICT or LOO_DF is None):
    print("⚠️  OPTIMIZE_PARAMS=True mas GT não disponível — otimização ignorada.")

# ============================================================================
# ADICIONADO: Mapeamento de colunas do Excel Extractrion_for_agents.xlsx
# ============================================================================
# Renomear coluna 'Code' para 'Manuel_de_Codage' (esperado pelo código)
if 'Code' in df.columns and 'Manuel_de_Codage' not in df.columns:
    df['Manuel_de_Codage'] = df['Code']
    print('✅ Coluna Code → Manuel_de_Codage mapeada')

# Preservar coluna 'translation' (análise do especialista) para comparações futuras
if 'translation' in df.columns:
    df['expert_translation'] = df['translation']  # Análise do especialista
    n_translations = df['translation'].notna().sum()
    print(f'✅ Coluna translation preservada: {n_translations} análises de especialista')
else:
    df['expert_translation'] = None
    print('⚠️  Coluna translation no encontrada en Excel')
# ============================================================================


# Definición de separadores MdC
MDL_SEPARATORS = {'-': 'linear', ':': 'stack', '*': 'ligature'}



# ============================================================================
# V1.28: SISTEMA DE CATEGORIZAÇÃO DE SINAIS (corrigido após avaliação v1.27)
# ============================================================================
# Baseado em avaliação v1.27 (99 sequências, F1=72.77%)
# v1.27 mostrou: W24/R8 validação funcionou (-90%/-92% FP), mas I9/L1/I10 
# causaram +110 FP (sinais novos sem validação)

SIGN_CATEGORIES = {
    # Categoria 1: Fonogramas estritos - NUNCA são determinativos
    "phonogram_strict": {"G17", "N35", "S29"},
    
    # Categoria 2: Fonogramas primários - raramente determinativos (penalidade pesada)
    "phonogram_primary": {"M17", "D21", "G43", "V31", "V4", "V28", "U7"},
    
    # Categoria 3: Críticos de contexto - determinativos VÁLIDOS mas requerem contexto
    "context_vessel": {"W24", "W22", "W23"},  # Requer contexto de vasos/líquidos/oferendas
    "context_divine": {"R8", "R7"},  # Requer contexto divino/sagrado
    "context_vision": {"D4"},  # Requer contexto de visão/ver
    "context_bovine": {"F4"},  # Requer contexto de gado/touro
    "context_aquatic": {"K1"},  # Requer contexto de peixe/água
    "context_food": {"X1"},  # X1 como determinativo (não fonograma), requer contexto de comida
    
    # V1.28: NOVOS - Sinais que causaram problemas em v1.27
    "context_animal": {"I9", "I10", "I5", "I6"},  # Cobras/répteis - MUITO ambíguos (fonograma vs det)
    "context_insect": {"L1", "L2"},  # Escaravelho/besouros - sagrados mas também fonéticos
    "context_tool": {"U1", "U6"},  # Ferramentas - precisam contexto agricultura/ferramenta
    "context_body_action": {"D58", "D56"},  # Partes do corpo em ação - precisam contexto movimento
    
    # Categoria 4: Determinativos confiáveis - sempre válidos quando finais/quase-finais
    "reliable": {"A1", "B1", "A40", "Aa2", "Q5", "O49", "O31", "A24", "N1", 
                "E1", "I14", "G40", "T14", "P1"},
    
    # Categoria 5: Marcadores plurais - funcionais, não classificadores semânticos
    "plural_functional": {"Z1", "Z2", "Z3"},
}

# V1.28: Limiares ajustados com base em avaliação v1.27
# I9, I10, L1 causaram +87 FP em v1.27 - precisam limiares MUITO altos
CONTEXT_THRESHOLDS = {
    # v1.27 (sucessos mantidos)
    "W24": 0.75,  # -90% FP em v1.27 (20→2)
    "W22": 0.75,
    "W23": 0.75,
    "R8": 0.75,   # -92% FP em v1.27 (13→1)
    "R7": 0.75,
    "D4": 0.85,
    "F4": 0.85,
    "K1": 0.85,
    "X1": 0.85,
    
    # V1.28: NOVOS - Sinais problemáticos identificados em v1.27
    "I9": 0.90,   # CRÍTICO: +47 FP em v1.27 - cobra muito ambígua
    "I10": 0.90,  # CRÍTICO: +18 FP em v1.27 - cobra variante
    "L1": 0.85,   # CRÍTICO: +22 FP em v1.27 - escaravelho sagrado mas também fonético
    "L2": 0.85,   # Preventivo: mesma série que L1
    "I5": 0.88,   # Preventivo: outros répteis
    "I6": 0.88,
    "D58": 0.82,  # +8 FP em v1.27 - perna avançando
    "D56": 0.80,  # Preventivo: mesmo campo semântico
    "U1": 0.78,   # +8 FP em v1.27 - foice
    "U6": 0.78,   # Preventivo: mesma série
}

# V1.28: Campos semânticos expandidos para novos contextos
SEMANTIC_FIELDS = {
    # v1.27 (mantidos)
    "vessel": {"W24", "W22", "W23", "Aa1", "X4", "bread", "beer", "wine", "oil", "water", "offering"},
    "divine": {"R8", "R7", "A40", "B1", "C12", "god", "goddess", "temple", "sacred", "divine", "deity"},
    "vision": {"D4", "eye", "see", "watch", "look", "perceive", "observe"},
    "bovine": {"F4", "E1", "bull", "cow", "cattle", "bovine", "ox"},
    "aquatic": {"K1", "fish", "water", "river", "canal", "fishing"},
    "food": {"X1", "bread", "food", "offering", "meal", "eat"},
    
    # V1.28: NOVOS campos semânticos
    "animal": {"I9", "I10", "I5", "I6", "E1", "E23", "snake", "serpent", "cobra", "reptile", "animal", "beast"},
    "insect": {"L1", "L2", "beetle", "scarab", "sacred", "divine", "amulet", "religious", "holy"},
    "tool": {"U1", "U6", "T14", "T15", "sickle", "hoe", "tool", "agriculture", "harvest", "farming", "plow"},
    "body_action": {"D58", "D56", "D54", "leg", "foot", "walk", "go", "move", "advance", "travel"},
}

# V1.28: Lista de sinais recém-adicionados (v1.27) - penalidade conservadora
# Estes sinais são novos e não foram testados extensivamente
RECENTLY_ADDED_SIGNS = {
    # Série I (répteis) - PROBLEMÁTICOS em v1.27
    "I9", "I10", "I5", "I6",
    # Série L (insetos) - PROBLEMÁTICO em v1.27
    "L1", "L2",
    # Série D (corpo) - PROBLEMÁTICO em v1.27
    "D56", "D57", "D58",
    # Série U (agricultura) - PROBLEMÁTICO em v1.27
    "U1", "U6",
    # Séries F, N, T, P, O (funcionaram bem, mas penalidade leve)
    "F51", "F52", "F53",
    "N16", "N23", "N24", "N26", "N27", "N28", "N29",
    "T14", "T15", "T16", "T17", "T19", "T20",
    "P1", "P2", "P3", "P4", "P6", "P8",
    "O38", "O39", "O40", "O41", "O43", "O44", "O45",
    # Outros novos
    "Z6", "Z7", "A7", "A14", "A28", "A50", "E23", "E34",
    "S12", "V13", "V15", "W10", "W11", "X2", "X8", "R7",
}

# Para compatibilidade retroativa com código legado
NEVER_DETERMINATIVE_SIGNS = SIGN_CATEGORIES["phonogram_strict"]

# Commonly polyvalent / typically phonogram-preferred signs in this pipeline context
LIKELY_FONOGRAM_SIGNS = {
    "M17", "D21", "X1", "G43", "N35", "G17",
    "O70", "O4", "I9", "V31", "V4", "V28", "U7",
}

# Strong determinatives / markers that are often determinative even when non-final in formulaic sequences
STRONG_DETERMINATIVE_SIGNS = {
    "Z1", "Z2", "Z3",
    "Aa2",
    "O49", "O31",  # O1 REMOVIDO - análise mostrou 80% de erro (4 FP vs 1 TP)
    "D20",
    "W9",
    "A1", "B1", "N1",
    "A35", "A40",
    "A24",
    "Q5", "X4",
}


# --- Generic sign categories used by the deterministic scorer (to avoid overfitting to any small gold set) ---

# Plural markers: in this project they are treated as part of the output set when present (keep repetitions).
PLURAL_MARKERS = {"Z1", "Z2", "Z3"}

# Very common "place/settlement" final determinatives. Keep this small and grounded in sign categories,
# not in hand-coded full sequences.
FINAL_LOCATION_SIGNS = {"O49", "O1"}

# Common location/building modifiers that often precede a final location sign in the same block.
# (This is a small category list, not a gold-set template.)
LOCATION_MODIFIER_SIGNS = {"O31"}

# Signs that are often over-predicted when used purely semantically; require stronger evidence to appear in MAIN.
# V1.27: DEPRECATED - Use SIGN_CATEGORIES["context_*"] instead
# Mantido apenas para compatibilidade com código legado
CONTEXTUAL_ONLY_SIGNS = {"W24", "W22", "W23", "R8", "R7", "D4", "F4", "K1", "X1"}

def _is_plural(tok: str) -> bool:
    return tok in PLURAL_MARKERS

def _is_final_location(tok: str) -> bool:
    return tok in FINAL_LOCATION_SIGNS

def _is_location_modifier(tok: str) -> bool:
    return tok in LOCATION_MODIFIER_SIGNS

def normalize_mdc(s: str) -> str:
    """Normaliza cadena MdC eliminando espacios extra"""
    if not isinstance(s, str):
        return ''
    s = s.strip()
    s = re.sub(r'\s+', '', s)
    return s

def tokenize_mdc(mdc: str):
    """Tokeniza secuencia MdC en glifos y separadores"""
    mdc = normalize_mdc(mdc)
    if not mdc:
        return []
    
    tokens, buff = [], ''
    for ch in mdc:
        if ch in MDL_SEPARATORS:
            if buff:
                tokens.append(buff)
                buff = ''
            tokens.append(ch)
        else:
            buff += ch
    if buff:
        tokens.append(buff)
    
    # Expandir ligatures
    expanded = []
    for t in tokens:
        if '*' in t and t not in MDL_SEPARATORS:
            parts = t.split('*')
            for i, p in enumerate(parts):
                if p:
                    expanded.append(p)
                if i < len(parts) - 1:
                    expanded.append('*')
        else:
            expanded.append(t)
    
    return expanded

# ============================================================================
# MdC SEGMENTER (simple structural representation) + JSON utilities (v1.15+)
# ============================================================================
# Goal:
# - Provide a light structural view of MdC tokens (including separators '-', ':', '*')
# - Enable agents to reason about "end-of-unit" positions more reliably
# - Support JSON end-to-end outputs from agents (no regex parsing as source of truth)

# def build_mdc_structure(tokens: List[str]) -> Dict[str, Any]:
#     """
#     v1.20.1 override: build a lightweight structure where '-' DOES NOT create word blocks.

#     Rationale:
#       In MdC, '-' is a linear sign separator inside the same word/compound.
#       When the input has no explicit word separators (spaces), we should treat the whole
#       sequence as a single compound unit. This stabilizes suffix-block decisions.
#     """
#     def is_sep(t: str) -> bool:
#         return t in MDL_SEPARATORS

#     # Flat sign stream (token indices of non-separators)
#     signs: List[Dict[str, Any]] = []
#     for ti, t in enumerate(tokens):
#         if not is_sep(t):
#             signs.append({"token_index": ti, "sign": t})

#     # Group detection (stack/ligature) stays the same (':', '*')
#     groups: List[Dict[str, Any]] = []
#     i = 0
#     while i < len(tokens):
#         t = tokens[i]
#         if is_sep(t):
#             i += 1
#             continue

#         if i + 1 < len(tokens) and tokens[i + 1] in (":", "*"):
#             sep_type = tokens[i + 1]
#             chain_sign_token_indices = [i]
#             j = i
#             while j + 1 < len(tokens) and tokens[j + 1] == sep_type:
#                 if j + 2 < len(tokens) and not is_sep(tokens[j + 2]):
#                     chain_sign_token_indices.append(j + 2)
#                     j += 2
#                 else:
#                     break
#             if len(chain_sign_token_indices) >= 2:
#                 groups.append({
#                     "type": "stack" if sep_type == ":" else "ligature",
#                     "sign_token_indices": chain_sign_token_indices,
#                     "start_token_index": chain_sign_token_indices[0],
#                     "end_token_index": chain_sign_token_indices[-1],
#                     "end_sign_token_index": chain_sign_token_indices[-1],
#                 })
#             i = j + 1
#             continue

#         i += 1

#     # Single compound block: all sign token indices in order
#     all_sign_idx = [s["token_index"] for s in signs]
#     blocks: List[Dict[str, Any]] = []
#     token_index_to_block: Dict[int, int] = {}
#     if all_sign_idx:
#         blocks = [{
#             "sign_token_indices": all_sign_idx,
#             "start_token_index": int(all_sign_idx[0]),
#             "end_token_index": int(all_sign_idx[-1]),
#             "end_sign_token_index": int(all_sign_idx[-1]),
#         }]
#         for sti in all_sign_idx:
#             token_index_to_block[int(sti)] = 0

#     block_final_token_indices = [blocks[0]["end_sign_token_index"]] if blocks else []

#     final_sign_token_index = signs[-1]["token_index"] if signs else None

#     candidate_final_token_indices = set()
#     if final_sign_token_index is not None:
#         candidate_final_token_indices.add(int(final_sign_token_index))
#     for g in groups:
#         if g.get("end_sign_token_index") is not None:
#             candidate_final_token_indices.add(int(g["end_sign_token_index"]))

#     # last-N sign indices
#     last_sign_token_indices: List[int] = []
#     if signs:
#         N = 12
#         last_sign_token_indices = [int(s["token_index"]) for s in signs[-N:]]
#         candidate_final_token_indices.update(last_sign_token_indices)

#     # plural hints
#     plural_marker_indices: List[int] = []
#     plural_head_indices: List[int] = []
#     plural_set = {"Z1", "Z2", "Z3"}
#     token_to_signpos = {int(s["token_index"]): k for k, s in enumerate(signs)}

#     for s in signs:
#         if s["sign"] in plural_set:
#             pi = int(s["token_index"])
#             plural_marker_indices.append(pi)
#             sp = token_to_signpos.get(pi)
#             if sp is None:
#                 continue
#             for back in range(1, 4):
#                 if sp - back < 0:
#                     break
#                 hi = int(signs[sp - back]["token_index"])
#                 hs = signs[sp - back]["sign"]
#                 if hs not in plural_set:
#                     plural_head_indices.append(hi)
#                     candidate_final_token_indices.add(hi)
#                     break

#     def _dedup(seq: List[int]) -> List[int]:
#         seen=set()
#         out=[]
#         for x in seq:
#             if x in seen:
#                 continue
#             seen.add(x)
#             out.append(x)
#         return out

#     plural_marker_indices = _dedup(plural_marker_indices)
#     plural_head_indices = _dedup(plural_head_indices)
#     last_sign_token_indices = _dedup(last_sign_token_indices)

#     return {
#         "signs": signs,
#         "groups": groups,
#         "final_sign_token_index": final_sign_token_index,
#         "candidate_final_token_indices": sorted(candidate_final_token_indices),
#         "last_sign_token_indices": last_sign_token_indices,
#         "plural_marker_indices": plural_marker_indices,
#         "plural_head_indices": plural_head_indices,
#         "blocks": blocks,
#         "block_final_token_indices": block_final_token_indices,
#         "token_index_to_block": token_index_to_block,
#     }

def build_mdc_structure(tokens: List[str]) -> Dict[str, Any]:
    """
    Build a lightweight structure from tokenized MdC sequence.

    Notes:
    - `tokens` includes separators as standalone tokens: '-', ':', '*'
    - Positions used across the project are token indices (0-based), including separators.

    Added (v1.18.1):
    - micro-boundaries: last-N sign indices + plural-adjacent indices
    - plural hints: plural_marker_indices, plural_head_indices
    """
    def is_sep(t: str) -> bool:
        return t in MDL_SEPARATORS

    # Flat sign stream (token indices of non-separators)
    signs: List[Dict[str, Any]] = []
    for ti, t in enumerate(tokens):
        if not is_sep(t):
            signs.append({"token_index": ti, "sign": t})

    # Detect compound groups connected by ':' (stack) or '*' (ligature).
    # We treat continuous chains like A : B : C as one group, same for ligatures.
    groups: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if is_sep(t):
            i += 1
            continue

        # At a sign. Check if followed by ':' or '*' to form a group chain.
        if i + 1 < len(tokens) and tokens[i + 1] in (":", "*"):
            sep_type = tokens[i + 1]  # ':' or '*'
            chain_sign_token_indices = [i]
            j = i
            while j + 1 < len(tokens) and tokens[j + 1] == sep_type:
                if j + 2 < len(tokens) and not is_sep(tokens[j + 2]):
                    chain_sign_token_indices.append(j + 2)
                    j += 2
                else:
                    break
            if len(chain_sign_token_indices) >= 2:
                groups.append({
                    "type": "stack" if sep_type == ":" else "ligature",
                    "sign_token_indices": chain_sign_token_indices,
                    "start_token_index": chain_sign_token_indices[0],
                    "end_token_index": chain_sign_token_indices[-1],
                    "end_sign_token_index": chain_sign_token_indices[-1],
                })
            i = j + 1
            continue

        i += 1

    
    # --- Word-block segmentation (simple) ---
    # We split blocks primarily on '-' separators. Additionally, we allow some strong signs
    # to end a block early (useful for determinatives that close a semantic unit).
    BLOCK_END_SIGNS = {"O49", "O1", "O31"}  # conservative list; tune if needed

    blocks: List[Dict[str, Any]] = []
    current: List[int] = []

    def _flush_block():
        nonlocal current
        if current:
            blocks.append({
                "sign_token_indices": list(current),
                "start_token_index": int(current[0]),
                "end_token_index": int(current[-1]),
                "end_sign_token_index": int(current[-1]),
            })
            current = []

    def _next_sign_index(start_i: int) -> int:
        j = start_i + 1
        while j < len(tokens):
            if tokens[j] in MDL_SEPARATORS:
                # '-' is a hard block boundary anyway
                if tokens[j] == "-":
                    return -1
                j += 1
                continue
            return j
        return -1

    for ti, t in enumerate(tokens):
        if is_sep(t):
            if t == "-":
                _flush_block()
            continue
        # sign
        current.append(ti)
        # optional early block end after strong end signs when another sign follows (without '-')
        if t in BLOCK_END_SIGNS:
            nxt = _next_sign_index(ti)
            if nxt != -1:
                _flush_block()

    _flush_block()
    
    # # --- NEW: Detect overly granular blocks (many single-sign blocks) and collapse into one block ---
    # # This happens when '-' is being used as a sign separator and we end up with blocks of length 1.
    # if blocks:
    #     lens = [len(b.get("sign_token_indices", []) or []) for b in blocks]
    #     frac_single = sum(1 for L in lens if L == 1) / max(1, len(lens))

    #     if frac_single >= 0.60:  # tune 0.60 ~ 0.80
    #         all_idx = [s["token_index"] for s in signs]  # all sign token indices (no separators)
    #         if all_idx:
    #             blocks = [{
    #                 "sign_token_indices": all_idx,
    #                 "start_token_index": int(all_idx[0]),
    #                 "end_token_index": int(all_idx[-1]),
    #                 "end_sign_token_index": int(all_idx[-1]),
    #             }]

    block_final_token_indices = [b["end_sign_token_index"] for b in blocks] if blocks else []
    token_index_to_block: Dict[int, int] = {}
    for bi, b in enumerate(blocks):
        for sti in b.get("sign_token_indices", []) or []:
            token_index_to_block[int(sti)] = int(bi)


    # Final and candidate-final indices (end of sequence + group ends)
    final_sign_token_index = signs[-1]["token_index"] if signs else None
    candidate_final_token_indices = set()
    if final_sign_token_index is not None:
        candidate_final_token_indices.add(final_sign_token_index)
    for g in groups:
        #candidate_final_token_indices.add(g["end_sign_token_index"])
        gi = int(g["end_sign_token_index"])
        # Only treat as candidate-final if near the global end (within last 2 signs)
        if signs:
            last4 = {s["token_index"] for s in signs[-2:]}
            if gi in last4:
                candidate_final_token_indices.add(gi)

    # Micro-boundaries (heuristic): last-N signs often contain determinatives/plural clusters
    last_sign_token_indices: List[int] = []
    if signs:
        N = 8
        last_sign_token_indices = [s["token_index"] for s in signs[-N:]]
        candidate_final_token_indices.update(last_sign_token_indices)

    # Plural hints: detect Z1/Z2/Z3 and the closest non-separator heads around them
    plural_marker_indices: List[int] = []
    plural_head_indices: List[int] = []
    plural_set = {"Z1", "Z2", "Z3"}
    token_to_signpos = {s["token_index"]: k for k, s in enumerate(signs)}

    for s in signs:
        if s["sign"] in plural_set:
            pi = int(s["token_index"])
            plural_marker_indices.append(pi)
            sp = token_to_signpos.get(pi)
            if sp is None:
                continue
            # head: previous non-plural sign within 3 positions
            for back in range(1, 4):
                if sp - back < 0:
                    break
                hi = int(signs[sp - back]["token_index"])
                hs = signs[sp - back]["sign"]
                if hs not in plural_set:
                    plural_head_indices.append(hi)
                    candidate_final_token_indices.add(hi)
                    break

    # Dedup keep order
    def _dedup(seq: List[int]) -> List[int]:
        seen=set()
        out=[]
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    plural_marker_indices = _dedup(plural_marker_indices)
    plural_head_indices = _dedup(plural_head_indices)
    last_sign_token_indices = _dedup(last_sign_token_indices)

    return {
        "signs": signs,
        "groups": groups,
        "final_sign_token_index": final_sign_token_index,
        "candidate_final_token_indices": sorted(candidate_final_token_indices),
        "last_sign_token_indices": last_sign_token_indices,
        "plural_marker_indices": plural_marker_indices,
        "plural_head_indices": plural_head_indices,
        "blocks": blocks,
        "block_final_token_indices": block_final_token_indices,
        "token_index_to_block": token_index_to_block,
    }


# def build_word_boundaries(tokens: List[str], mdc_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
#     """
#     v1.20.1 override: by default treat the whole sequence as ONE compound word.
#     This matches MdC usage when inputs don't include explicit word separators.
#     """
#     all_sign_idx = [i for i, t in enumerate(tokens) if t not in MDL_SEPARATORS]
#     if not all_sign_idx:
#         return []
#     signs = [tokens[i] for i in all_sign_idx]
#     return [{
#         "word_id": 0,
#         "sign_token_indices": all_sign_idx,
#         "final_token_index": int(all_sign_idx[-1]),
#         "text": "-".join(signs),
#     }]


def build_word_boundaries(tokens: List[str], mdc_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a lightweight list of probable word/compound units from MdC tokens.

    Rationale (Egyptological practice):
    Determinatives are usually final of the *word/compound*, not necessarily the whole sequence.
    We approximate words using the precomputed 'blocks' (split on '-') plus group hints.

    Output:
      [
        {
          "word_id": int,
          "sign_token_indices": [int, ...],     # token indices of signs (no separators)
          "final_token_index": int or None,     # last sign token index
          "text": "A1-N1-... (signs only, '-')",
        }, ...
      ]
    """
    blocks = (mdc_structure or {}).get("blocks", []) or []
    words: List[Dict[str, Any]] = []

    for wi, b in enumerate(blocks):
        sti = [int(x) for x in (b.get("sign_token_indices") or []) if isinstance(x, (int, float, str)) and str(x).isdigit()]
        if not sti:
            continue
        signs = [tokens[i] for i in sti if 0 <= i < len(tokens) and tokens[i] not in MDL_SEPARATORS]
        if not signs:
            continue
        final_idx = int(sti[-1])
        words.append({
            "word_id": int(wi),
            "sign_token_indices": sti,
            "final_token_index": final_idx,
            "text": "-".join(signs),
        })

    # Fallback: if no blocks, treat whole sequence as one word of all signs
    if not words:
        all_sign_idx = [i for i, t in enumerate(tokens) if t not in MDL_SEPARATORS]
        if all_sign_idx:
            signs = [tokens[i] for i in all_sign_idx]
            words = [{
                "word_id": 0,
                "sign_token_indices": all_sign_idx,
                "final_token_index": int(all_sign_idx[-1]),
                "text": "-".join(signs),
            }]
    return words




def generate_suffix_span_candidates(words: List[Dict[str, Any]], tokens: List[str], max_len: int = 6) -> List[Dict[str, Any]]:
    """
    Generate contiguous suffix span candidates (determinative-block candidates) per word/compound.

    Egyptological alignment:
      - Determinatives are typically FINAL of the word/compound.
      - We therefore consider ONLY suffix spans (contiguous) of each inferred word unit.
      - Repetitions are preserved (each token index is kept).

    Output format (stored in state as `word_span_candidates`):
      [
        {
          "word_id": 0,
          "candidates": [
             {"span_token_indices": [...], "text": "Z1-A1", "length": 2},
             {"span_token_indices": [...], "text": "", "length": 0},  # empty span
             ...
          ]
        }, ...
      ]
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(words, list):
        return out
    for w in words:
        if not isinstance(w, dict):
            continue
        wid = int(w.get("word_id", 0))
        idxs = w.get("sign_token_indices") or []
        idxs = [int(i) for i in idxs if isinstance(i, (int, float)) or (isinstance(i, str) and str(i).isdigit())]
        idxs = [i for i in idxs if 0 <= i < len(tokens) and tokens[i] not in MDL_SEPARATORS]
        if not idxs:
            continue
        max_k = min(max_len, len(idxs))
        cands: List[Dict[str, Any]] = []
        # include empty span as a valid option (sem determinativos para esta palavra)
        cands.append({"span_token_indices": [], "text": "", "length": 0})
        for k in range(1, max_k + 1):
            span = idxs[-k:]  # suffix
            span_tokens = [tokens[i] for i in span]
            cands.append({
                "span_token_indices": span,
                "text": "-".join(span_tokens),
                "length": k
            })
        out.append({"word_id": wid, "candidates": cands})
    return out


def build_phonetic_complement_hints(tokens: List[str], words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute lightweight hints for likely phonetic complements.

    Egyptological rationale:
      - In Middle Egyptian, common uniliteral phonograms are frequently used as *phonetic complements*,
        especially near the end of a word/compound. They should NOT be treated as determinatives by default.

    Heuristic (conservative):
      - Look at the last 3 sign tokens of each inferred word/compound.
      - If a token is in LIKELY_FONOGRAM_SIGNS or RARELY_DETERMINATIVES, and is NOT a strong determinative nor a plural marker,
        mark it as complement-likely.
      - These are HINTS only; they should reduce false positives, not block true determinatives.

    Returns:
      [{"token": "M17", "index": 12, "word_id": 0, "reason": "..."}]
    """
    hints: List[Dict[str, Any]] = []
    if not isinstance(tokens, list) or not isinstance(words, list):
        return hints
    for w in words:
        if not isinstance(w, dict):
            continue
        wid = int(w.get("word_id", 0))
        idxs = w.get("sign_token_indices") or []
        idxs = [int(i) for i in idxs if isinstance(i, (int, float)) or (isinstance(i, str) and str(i).isdigit())]
        idxs = [i for i in idxs if 0 <= i < len(tokens) and tokens[i] not in MDL_SEPARATORS]
        tail = idxs[-3:] if len(idxs) >= 1 else []
        for idx in tail:
            tok = tokens[idx]
            if tok in STRONG_DETERMINATIVE_SIGNS or tok in PLURAL_MARKERS:
                continue
            if tok in LIKELY_FONOGRAM_SIGNS or tok in RARELY_DETERMINATIVES:
                hints.append({
                    "token": tok,
                    "index": int(idx),
                    "word_id": wid,
                    "reason": "Common uniliteral/phonogram near word end → likely phonetic complement (not determinative by default)."
                })
    # Deduplicate by (token,index)
    seen=set()
    out=[]
    for h in hints:
        key=(h["token"], int(h["index"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _extract_json_substring(s: str) -> str:
    """Best-effort extraction of a single JSON object inside a string.

    Supports optional BEGIN_JSON/END_JSON markers. Falls back to balanced-brace extraction.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip()

    begin = "BEGIN_JSON"
    endm = "END_JSON"
    if begin in s and endm in s:
        try:
            inner = s.split(begin, 1)[1].split(endm, 1)[0].strip()
            inner = re.sub(r"^```(?:json)?\s*|\s*```$", "", inner.strip(),
                           flags=re.IGNORECASE | re.DOTALL)
            return inner
        except Exception:
            pass

    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(),
               flags=re.IGNORECASE | re.DOTALL)

    if "{" not in s or "}" not in s:
        return ""

    start = s.find("{")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":  # backslash
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]

    start = s.find("{")
    end = s.rfind("}") + 1
    return s[start:end] if start != -1 and end > start else ""

def parse_llm_json(content: str) -> Dict[str, Any]:
    """
    Parse JSON returned by an LLM.
    Returns {} on failure (never raises).
    """
    if not isinstance(content, str) or not content.strip():
        return {}
    js = _extract_json_substring(content)
    if not js:
        return {}

    def _repair(x: str) -> str:
        x = x.replace("“", '"').replace("”", '"').replace("’", "'")
        x = re.sub(r",\s*([}\]])", r"\1", x)
        return x.strip()

    for attempt in range(2):
        try:
            data = json.loads(_repair(js))
            return data if isinstance(data, dict) else {}
        except Exception:
            if attempt == 0:
                js = js.strip()
                continue
    return {}

def get_candidates_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract determinative candidates from LLM JSON across supported schema variants.

    The project evolved across versions; this helper makes logs and downstream logic robust.
    """
    if not isinstance(data, dict):
        return []
    # v3+: nested under main_interpretation
    mi = data.get("main_interpretation")
    if isinstance(mi, dict):
        dets = mi.get("determinatives") or mi.get("determinative_spans") or mi.get("determinative_candidates")
        if isinstance(dets, list):
            return dets
    # v2/v1: top-level determinatives
    dets = data.get("determinatives") or data.get("candidates")
    if isinstance(dets, list):
        return dets
    return []


def normalize_determinatives_from_json(
    det_list: Any,
    tokens: List[str],
    default_method: str = "llm_json",
    determinatives_dict: Dict = None,
    state: Dict = None
) -> List[Dict[str, Any]]:
    """
    Convert a JSON list of determinatives into the project's standard format:
      {sign, position, significado, categoria, semantica, confidence, method, ...}

    Rules:
    - index is 0-based token index (including separators).
    - token must match tokens[index] and must NOT be a separator.
    - Unknown signs are kept with minimal metadata.
    - confidence is calculated based on multi-source evidence if state is provided.
    """
    if not isinstance(det_list, list):
        return []
    
    # Build evidence map from state for confidence calculation
    evidence_map = {}  # key: (token, index), value: set of sources
    if state:
        # From semantic
        sem_json = state.get('semantic_json', {})
        sem_dets = get_candidates_from_json(sem_json)
        for d in sem_dets:
            if isinstance(d, dict):
                tok = d.get('token') or d.get('sign')
                idx = d.get('index') or d.get('position')
                if tok and idx is not None:
                    try:
                        idx = int(idx)
                        key = (tok, idx)
                        if key not in evidence_map:
                            evidence_map[key] = set()
                        evidence_map[key].add('semantic')
                    except: pass
        
        # From structural
        struct_json = state.get('structural_json', {})
        struct_dets = get_candidates_from_json(struct_json)
        for d in struct_dets:
            if isinstance(d, dict):
                tok = d.get('token') or d.get('sign')
                idx = d.get('index') or d.get('position')
                if tok and idx is not None:
                    try:
                        idx = int(idx)
                        key = (tok, idx)
                        if key not in evidence_map:
                            evidence_map[key] = set()
                        evidence_map[key].add('structural')
                    except: pass
        
        # From rules
        rule_result = state.get('rule_result', {})
        rule_dets = rule_result.get('determinatives', [])
        for d in rule_dets:
            if isinstance(d, dict):
                tok = d.get('sign')
                idx = d.get('position')
                if tok and idx is not None:
                    try:
                        idx = int(idx)
                        key = (tok, idx)
                        if key not in evidence_map:
                            evidence_map[key] = set()
                        evidence_map[key].add('rules')
                    except: pass

    out = []
    for d in det_list:
        if not isinstance(d, dict):
            continue

        token = (d.get("token") or d.get("sign") or "").strip()
        idx = d.get("index", d.get("token_index", d.get("position", None)))
        try:
            idx = int(idx)
        except Exception:
            continue

        if not token or not (0 <= idx < len(tokens)):
            continue
        if tokens[idx] != token:
            continue
        if token in MDL_SEPARATORS:
            continue
        if token in NEVER_DETERMINATIVE_SIGNS:
            continue

        # Calculate confidence from provided value or multi-source evidence
        conf = d.get("confidence", d.get("score", None))
        if conf is None and evidence_map:
            # Calculate based on evidence from multiple sources
            key = (token, idx)
            sources = evidence_map.get(key, set())
            num_sources = len(sources)
            
            # Confidence formula based on number of confirming sources
            if num_sources >= 3:
                conf = 0.9  # All three sources agree (semantic + structural + rules)
            elif num_sources == 2:
                conf = 0.75  # Two sources agree
            elif num_sources == 1:
                conf = 0.6  # Only one source (in addition to synth)
            else:
                conf = 0.4  # Only synthesizer (no other evidence)
        
        if conf is None:
            conf = 0.0
            
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0

        det_dict = determinatives_dict if determinatives_dict is not None else DETERMINATIVES_DETAILED
        info = det_dict.get(token, {})
        out.append({
            "sign": token,
            "position": idx,
            "significado": info.get("significado", info.get("meaning", "Unknown")),
            "categoria": info.get("categoria", info.get("category", "unknown")),
            "semantica": info.get("semantica", info.get("semantic_field", "unknown")),
            "confidence": conf,
            "method": d.get("method_final") or d.get("method") or default_method,
            "rationale": d.get("rationale") or d.get("justification") or d.get("evidence") or "",
            "needs_lookup": bool(d.get("needs_lookup", False)),
            "word_id": d.get("word_id", None),
            "det_type": d.get("det_type") or d.get("type") or "",
            "alt_codes": d.get("alt_codes", []),
        })

    # Deduplicate exact (sign, position), keep max confidence
    dedup = {}
    for item in out:
        key = (item["sign"], item["position"])
        if key not in dedup or item["confidence"] > dedup[key]["confidence"]:
            dedup[key] = item
    return list(dedup.values())




# ============================================================================
# CONSOLIDATION SCORER (deterministic) — improves MAIN vs ALTERNATIVES selection
# ============================================================================
def _candidate_key(token: str, idx: int) -> tuple:
    return (token, int(idx))

def _gather_candidate_evidence(state: Dict[str, Any]) -> Dict[tuple, Dict[str, Any]]:
    """Collect candidate determinatives from semantic_json, structural_json and rule_result.

    Added (v1.18.1):
    - structural hints from build_mdc_structure: near-end, plural markers/heads
    """
    pool: Dict[tuple, Dict[str, Any]] = {}

    tokens = state.get("tokens") or []
    mdc = state.get("mdc_structure", {}) or {}
    block_final = set(mdc.get("block_final_token_indices", []) or [])
    global_final = mdc.get("final_sign_token_index", None)
    group_ends = set(int(g.get("end_sign_token_index")) for g in (mdc.get("groups") or []) if isinstance(g, dict) and g.get("end_sign_token_index") is not None)
    token_to_block = (mdc.get("token_index_to_block") or {})
    near_end = set(mdc.get("last_sign_token_indices", []) or [])
    plural_markers = set(mdc.get("plural_marker_indices", []) or [])
    plural_heads = set(mdc.get("plural_head_indices", []) or [])

    def _touch(token: str, idx: int) -> Dict[str, Any]:
        key = _candidate_key(token, idx)
        if key not in pool:
            pool[key] = {
                "token": token,
                "index": idx,
                "sources": set(),
                "conf_sem": 0.0,
                "conf_struct": 0.0,
                "conf_rule": 0.0,
                "is_final": idx in block_final,
                "is_global_final": (global_final is not None and idx == int(global_final)),
                "is_group_end": idx in group_ends,
                "block_id": int(token_to_block.get(idx, -1)) if isinstance(token_to_block, dict) else -1,
                "is_near_end": idx in near_end,
                "is_plural_marker": idx in plural_markers,
                "is_plural_head": idx in plural_heads,
                "is_phonetic_complement_likely": False,
            }
        # Mark complement-likely occurrences (hints)
        if (token, int(idx)) in phon_pairs:
            pool[key]["is_phonetic_complement_likely"] = True
        return pool[key]

    phon_pairs = state.get("phonetic_complement_pairs") or set()
    if not isinstance(phon_pairs, set):
        try:
            phon_pairs = set(tuple(x) for x in phon_pairs)
        except Exception:
            phon_pairs = set()

    def _valid_tok(tok: str, idx: int) -> bool:
        if not tok or not (0 <= idx < len(tokens)):
            return False
        if tokens[idx] != tok:
            return False
        if tok in MDL_SEPARATORS:
            return False
        return True

    # Semantic candidates
    sem = (state.get("semantic_json") or {}).get("determinatives", []) or []
    if isinstance(sem, list):
        for d in sem:
            if not isinstance(d, dict):
                continue
            tok = (d.get("token") or "").strip()
            idx = d.get("index", None)
            try:
                idx = int(idx)
            except Exception:
                continue
            if not _valid_tok(tok, idx):
                continue
            item = _touch(tok, idx)
            item["sources"].add("semantic")
            try:
                item["conf_sem"] = max(item["conf_sem"], float(d.get("confidence", 0.0)))
            except Exception:
                pass

    # Structural candidates
    stc = (state.get("structural_json") or {}).get("determinatives", []) or []
    if isinstance(stc, list):
        for d in stc:
            if not isinstance(d, dict):
                continue
            tok = (d.get("token") or "").strip()
            idx = d.get("index", None)
            try:
                idx = int(idx)
            except Exception:
                continue
            if not _valid_tok(tok, idx):
                continue
            item = _touch(tok, idx)
            item["sources"].add("structural")
            try:
                item["conf_struct"] = max(item["conf_struct"], float(d.get("confidence", 0.0)))
            except Exception:
                pass

    # Rule-based candidates
    rr = (state.get("rule_result") or {}).get("determinatives", []) or []
    if isinstance(rr, list):
        for d in rr:
            if not isinstance(d, dict):
                continue
            tok = (d.get("sign") or "").strip()
            idx = d.get("position", None)
            try:
                idx = int(idx)
            except Exception:
                continue
            if not _valid_tok(tok, idx):
                continue
            item = _touch(tok, idx)
            item["sources"].add("rules")
            conf_str = (d.get("confidence") or "").strip().lower()
            if conf_str == "high":
                item["conf_rule"] = max(item["conf_rule"], 1.0)
            elif conf_str == "medium":
                item["conf_rule"] = max(item["conf_rule"], 0.7)
            else:
                item["conf_rule"] = max(item["conf_rule"], 0.4)

    return pool


# ============================================================================
# V1.27: VALIDAÇÃO DE CONTEXTO SEMÂNTICO
# ============================================================================

def validate_semantic_context(sign: str, sequence_tokens: list, semantic_hints: str = "") -> tuple:
    """
    V1.28: Valida se um sinal aparece em contexto semântico apropriado.
    Expandido para incluir animal, insect, tool, body_action (v1.27 problem signs).
    
    Args:
        sign: O sinal a validar (ex: "W24", "R8", "I9", "L1")
        sequence_tokens: Lista de tokens da sequência
        semantic_hints: Dicas semânticas do agente semântico (opcional)
    
    Returns:
        (has_context: bool, context_strength: float)
        - has_context: True se contexto apropriado foi encontrado
        - context_strength: 0.0-1.0, força do contexto detectado
    """
    # Determinar qual campo semântico o sinal pertence
    sign_field = None
    if sign in SIGN_CATEGORIES.get("context_vessel", set()):
        sign_field = "vessel"
    elif sign in SIGN_CATEGORIES.get("context_divine", set()):
        sign_field = "divine"
    elif sign in SIGN_CATEGORIES.get("context_vision", set()):
        sign_field = "vision"
    elif sign in SIGN_CATEGORIES.get("context_bovine", set()):
        sign_field = "bovine"
    elif sign in SIGN_CATEGORIES.get("context_aquatic", set()):
        sign_field = "aquatic"
    elif sign in SIGN_CATEGORIES.get("context_food", set()):
        sign_field = "food"
    # V1.28: Novos campos para sinais problemáticos
    elif sign in SIGN_CATEGORIES.get("context_animal", set()):
        sign_field = "animal"
    elif sign in SIGN_CATEGORIES.get("context_insect", set()):
        sign_field = "insect"
    elif sign in SIGN_CATEGORIES.get("context_tool", set()):
        sign_field = "tool"
    elif sign in SIGN_CATEGORIES.get("context_body_action", set()):
        sign_field = "body_action"
    
    if not sign_field:
        # Não é um sinal crítico de contexto
        return True, 1.0
    
    # Obter sinais/palavras do campo semântico
    field_signs = SEMANTIC_FIELDS.get(sign_field, set())
    
    # Contar correspondências na sequência
    matches = 0
    for tok in sequence_tokens:
        if tok in field_signs:
            matches += 1
    
    # Verificar também nas dicas semânticas do LLM
    if semantic_hints:
        semantic_lower = str(semantic_hints).lower()
        for keyword in field_signs:
            if isinstance(keyword, str) and keyword.lower() in semantic_lower:
                matches += 2  # Dicas do LLM têm peso maior
                break
    
    # V1.28: Calcular força do contexto com boost para sinais críticos (I9, L1)
    # Sinais com limiares altos (>0.85) precisam evidência mais forte
    threshold = CONTEXT_THRESHOLDS.get(sign, 0.75)
    if threshold >= 0.85:
        # Sinais muito críticos (I9, I10, L1) - requerem mais matches
        context_strength = min(1.0, matches * 0.25)  # Mais conservador
        has_context = context_strength >= 0.6  # Limiar mais alto
    else:
        # Sinais moderadamente críticos (W24, R8, etc)
        context_strength = min(1.0, matches * 0.3)
        has_context = context_strength >= 0.5
    
    return has_context, context_strength


def get_sign_category(sign: str) -> str:
    """Retorna a categoria de um sinal."""
    for cat_name, signs in SIGN_CATEGORIES.items():
        if sign in signs:
            return cat_name
    return "unknown"



def _score_candidate(item: Dict[str, Any]) -> float:
    """
    V1.28: Função de scoring consciente de contexto e categorias.
    Corrigido após avaliação v1.27 - adiciona validação para I9, L1, I10
    e penalidade para sinais recém-adicionados.
    """
    tok = item.get("token") or ""
    #idx = item.get("index", -1)
    is_final = bool(item.get("is_final", False))
    is_global_final = bool(item.get("is_global_final", False))
    is_group_end = bool(item.get("is_group_end", False))
    is_near_end = bool(item.get("is_near_end", False))
    is_plural_marker = bool(item.get("is_plural_marker", False))
    is_plural_head = bool(item.get("is_plural_head", False))
    sources = item.get("sources") or set()
    src_n = len(sources) if isinstance(sources, set) else 0

    # Categoria 1: Fonogramas estritos - bloqueio absoluto
    if tok in SIGN_CATEGORIES.get("phonogram_strict", set()):
        return 0.0

    # Base evidence weights
    score = 0.0
    score += 0.38 * float(item.get("conf_sem", 0.0))
    score += 0.34 * float(item.get("conf_struct", 0.0))
    score += 0.28 * float(item.get("conf_rule", 0.0))

    # Multi-source agreement bonuses
    if src_n >= 2:
        score += 0.08
    if src_n >= 3:
        score += 0.06

    # Categoria 2: Fonogramas primários - penalidade pesada (não bloqueio)
    if tok in SIGN_CATEGORIES.get("phonogram_primary", set()):
        score *= 0.4  # Reduz para 40% a menos que haja evidência muito forte

    # Categoria 3: Sinais críticos de contexto - validação de campo semântico
    # V1.28: Expandido para incluir animal, insect, tool, body_action
    is_context_critical = False
    for cat_name in ["context_vessel", "context_divine", "context_vision", 
                     "context_bovine", "context_aquatic", "context_food",
                     "context_animal", "context_insect", "context_tool", "context_body_action"]:
        if tok in SIGN_CATEGORIES.get(cat_name, set()):
            is_context_critical = True
            break
    
    if is_context_critical:
        # Validar contexto semântico
        sequence_tokens = item.get("sequence_tokens", [])
        semantic_hints = item.get("semantic_hints", "")
        has_context, context_strength = validate_semantic_context(tok, sequence_tokens, semantic_hints)
        
        if not has_context:
            # Sem contexto apropriado: aplicar limiar ou rejeitar
            threshold = CONTEXT_THRESHOLDS.get(tok, 0.80)
            if score < threshold:
                return 0.0  # Rejeitar
            else:
                score *= 0.6  # Rebaixar mas manter
        else:
            # Com contexto: aumentar confiança
            score *= (1.0 + context_strength * 0.2)

    # V1.28: NOVO - Penalidade para sinais recém-adicionados
    # Sinais adicionados em v1.27 precisam evidência mais forte (causa de +110 FP)
    if tok in RECENTLY_ADDED_SIGNS:
        # Penalidade mais forte para sinais que causaram problemas em v1.27
        if tok in {"I9", "I10", "L1"}:  # Maiores culpados
            score *= 0.60  # 40% de penalidade
        elif tok in {"D58", "U1", "L2", "I5", "I6"}:  # Problemas menores
            score *= 0.75  # 25% de penalidade
        else:  # Outros sinais novos (funcionaram bem mas ainda não testados)
            score *= 0.85  # 15% de penalidade preventiva

    # Categoria 4: Determinativos confiáveis - bônus
    if tok in SIGN_CATEGORIES.get("reliable", set()):
        score += 0.12

    # Categoria 5: Marcadores plurais - bônus
    if tok in SIGN_CATEGORIES.get("plural_functional", set()):
        if is_plural_marker or is_near_end:
            score += 0.05
        if is_plural_head:
            score += 0.07

    # Bônus posicionais
    if is_final:
        score += 0.10
    if is_group_end and not is_final:
        score += 0.04
    if is_global_final:
        score += 0.06
    if is_near_end and not is_final:
        score += 0.06

    # Compatibilidade: bônus para STRONG_DETERMINATIVE_SIGNS (deprecated mas mantido)
    if tok in STRONG_DETERMINATIVE_SIGNS and tok not in SIGN_CATEGORIES.get("reliable", set()):
        score += 0.08

    # Penalidade de complemento fonético
    if bool(item.get('is_phonetic_complement_likely', False)):
        score -= 0.15 if src_n <= 1 else 0.06

    # Penalidade adicional para fonogramas primários não-finais
    if tok in SIGN_CATEGORIES.get("phonogram_primary", set()) and not is_final:
        score -= 0.35 if src_n <= 1 else 0.18

    # Clamp
    return max(0.0, min(1.0, score))




def _select_main_and_alts_from_scores(state: Dict[str, Any]) -> Dict[str, Any]:
    """Select MAIN + ALTERNATIVES using **contiguous suffix spans per word/compound**.

    Key structural change (v1.19+):
      - Determinatives are selected as *determinative blocks* (suffix spans) per inferred word/compound.
      - We then combine per-word selections via a small beam search to form the global determinative set.

    Output:
      {"main_dets": [...], "alt_sets": [[...], ...], "main_per_word": [...], "alt_per_word": [[...], ...]}
    """
    tokens: List[str] = state.get("tokens") or []
    words: List[Dict[str, Any]] = state.get("words") or []
    mdc = state.get("mdc_structure", {}) or {}

    # Fallback if word boundaries missing
    if not words:
        words = build_word_boundaries(tokens, mdc)

    pool = _gather_candidate_evidence(state)

    # Per-occurrence score lookup
    occ_score: Dict[tuple, float] = {}
    for key, item in pool.items():
        occ_score[key] = _score_candidate(item)

    def _as_det_dict(tok: str, idx: int, wid: int) -> Dict[str, Any]:
        return {
            "token": tok,
            "index": int(idx),
            "gardiner_code": None,
            "needs_lookup": (tok not in DETERMINATIVES_DETAILED),
            "word_id": int(wid),
        }

    def _canonical_key(det_list: List[Dict[str, Any]]) -> tuple:
        pairs = []
        for d in det_list or []:
            if not isinstance(d, dict):
                continue
            tok = (d.get("token") or d.get("sign") or "").strip()
            idx = d.get("index", d.get("position", None))
            try:
                idx = int(idx)
            except Exception:
                continue
            if tok and 0 <= idx < len(tokens) and tokens[idx] == tok and tok not in MDL_SEPARATORS and tok not in NEVER_DETERMINATIVE_SIGNS:
                pairs.append((tok, idx))
        # keep duplicates by index ordering (repetitions matter), but canonical key needs unique to compare sets:
        return tuple(sorted(set(pairs), key=lambda x: x[1]))

    def _score_span(span_token_indices: List[int]) -> float:
        # Sum token-level scores + light priors; DO NOT add tokens outside the span.
        if not span_token_indices:
            # empty determinatives are allowed and should be competitive when evidence is weak
            return 0.0
        s = 0.0
        span_tokens = []
        for idx in span_token_indices:
            if not (0 <= idx < len(tokens)):
                continue
            tok = tokens[idx]
            span_tokens.append(tok)
            s += occ_score.get(_candidate_key(tok, idx), 0.0)
            if tok in STRONG_DETERMINATIVE_SIGNS:
                s += 0.03
        # Structural priors for common determinative-block shapes
        if any(t in PLURAL_MARKERS for t in span_tokens):
            s += 0.05
        if any(_is_final_location(t) for t in span_tokens):
            s += 0.06
        if any(_is_location_modifier(t) for t in span_tokens) and any(_is_final_location(t) for t in span_tokens):
            s += 0.04
        # Penalize contextual-only signs unless span is strong (keeps W24/D4 etc. out of MAIN unless supported)
        if any(t in CONTEXTUAL_ONLY_SIGNS for t in span_tokens):
            s -= 0.25
        # Clamp
        return max(0.0, min(3.0, s))

    # Build per-word suffix span candidates (already computed in state when possible)
    span_cands = state.get("word_span_candidates")
    if not isinstance(span_cands, list) or not span_cands:
        span_cands = generate_suffix_span_candidates(words, tokens, max_len=6)

    # Map word_id -> list[span_indices]
    per_word_spans: Dict[int, List[List[int]]] = {}
    for w in span_cands:
        if not isinstance(w, dict):
            continue
        wid = int(w.get("word_id", 0))
        cands = w.get("candidates") or []
        spans = []
        for c in cands:
            if not isinstance(c, dict):
                continue
            span = c.get("span_token_indices") or []
            if not isinstance(span, list):
                continue
            # validate span: indices in range, no separators, no NEVER
            ok = True
            for idx in span:
                try:
                    idx = int(idx)
                except Exception:
                    ok = False
                    break
                if not (0 <= idx < len(tokens)):
                    ok = False
                    break
                tok = tokens[idx]
                if tok in MDL_SEPARATORS or tok in NEVER_DETERMINATIVE_SIGNS:
                    ok = False
                    break
            if ok:
                spans.append([int(i) for i in span])
        # dedup spans
        uniq=[]
        seen=set()
        for sp in spans:
            key=tuple(sp)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(sp)
        per_word_spans[wid] = uniq if uniq else [[]]

    # Beam search over words (keep best few combinations)
    BEAM = 10
    beam: List[Dict[str, Any]] = [{"score": 0.0, "det_list": [], "per_word": []}]  # per_word stores chosen span per word

    # Ensure deterministic order by word_id
    word_order = sorted([int(w.get("word_id", 0)) for w in words if isinstance(w, dict)], key=lambda x: x)
    word_lookup = {int(w.get("word_id", 0)): w for w in words if isinstance(w, dict)}

    for wid in word_order:
        spans = per_word_spans.get(wid, [[]])
        w = word_lookup.get(wid, {})
        new_beam = []
        for entry in beam:
            base_score = float(entry["score"])
            base_det = entry["det_list"]
            base_pw = entry["per_word"]
            for sp in spans:
                sp_score = _score_span(sp)
                dets = [ _as_det_dict(tokens[idx], idx, wid) for idx in sp ]  # preserve repetitions
                combined = base_det + dets
                new_beam.append({
                    "score": base_score + sp_score,
                    "det_list": combined,
                    "per_word": base_pw + [{"word_id": wid, "span_token_indices": sp, "text": "-".join([tokens[i] for i in sp])}],
                })
        # keep top BEAM
        new_beam.sort(key=lambda x: (-x["score"], len(_canonical_key(x["det_list"])) ))
        beam = new_beam[:BEAM]

    # Rank final combos and extract MAIN + ALTS
    if not beam:
        return {"main_dets": [], "alt_sets": []}

    # Create ranked unique by set key
    best_by_key: Dict[tuple, Dict[str, Any]] = {}
    for e in beam:
        k = _canonical_key(e["det_list"])
        if k not in best_by_key or e["score"] > best_by_key[k]["score"]:
            best_by_key[k] = e
    ranked = sorted(best_by_key.values(), key=lambda x: (-x["score"], len(_canonical_key(x["det_list"])) ))

    main_entry = ranked[0]
    main = main_entry["det_list"]
    main_key = _canonical_key(main)
    alt_sets: List[List[Dict[str, Any]]] = []
    alt_per_word: List[List[Dict[str, Any]]] = []
    for e in ranked[1:]:
        k = _canonical_key(e["det_list"])
        if k == main_key:
            continue
        alt_sets.append(e["det_list"])
        alt_per_word.append(e.get("per_word") or [])
        if len(alt_sets) >= 3:
            break

    return {
        "main_dets": main,
        "alt_sets": alt_sets,
        "main_per_word": main_entry.get("per_word") or [],
        "alt_per_word": alt_per_word,
    }


def postprocess_synth_json_with_scoring(synth_data: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(synth_data, dict):
        return synth_data

    tokens = state.get("tokens") or []
    mdc_struct = state.get("mdc_structure") or {}
    final_idxs = set(mdc_struct.get("candidate_final_token_indices") or [])
    if not final_idxs and tokens:
        # fallback: considera o último token como final
        final_idxs = {len(tokens) - 1}

    def _is_final(idx: int) -> bool:
        return idx in final_idxs

    # primeiro calcula o scored_set (para permitir exceções quando o scorer confirmou)
    sel = _select_main_and_alts_from_scores(state)
    scored_main = sel["main_dets"]
    scored_set = {(d.get("token"), int(d.get("index"))) for d in (scored_main or []) if isinstance(d, dict) and d.get("token") is not None and d.get("index") is not None}

    def _filter_det_list(det_list: Any) -> List[Dict[str, Any]]:
        out = []
        if not isinstance(det_list, list):
            return out

        for d in det_list:
            if not isinstance(d, dict):
                continue
            tok = (d.get("token") or d.get("sign") or "").strip()
            idx = d.get("index", d.get("position", None))
            try:
                idx = int(idx)
            except Exception:
                continue

            if tok in NEVER_DETERMINATIVE_SIGNS:
                continue

            # ✅ NOVO: corta sinais contextuais/arriscados quando não estão em posição final,
            # EXCETO se o scorer também escolheu exatamente (tok, idx).
            if (tok in CONTEXTUAL_ONLY_SIGNS) and (not _is_final(idx)) and ((tok, idx) not in scored_set):
                continue

            # opcional: cortar fonogramas prováveis fora do final
            if tok in LIKELY_FONOGRAM_SIGNS and (not _is_final(idx)) and ((tok, idx) not in scored_set):
                continue

            out.append({
                "token": tok,
                "index": idx,
                "gardiner_code": d.get("gardiner_code"),
                "needs_lookup": bool(d.get("needs_lookup", False))
            })
        return out

    main = (synth_data.get("main_interpretation", {}) or {})
    llm_main = _filter_det_list(main.get("determinatives", []) or [])

    # ... (restante da função segue como já está, usando llm_main e scored_main)


    main = (synth_data.get("main_interpretation", {}) or {})
    llm_main = _filter_det_list(main.get("determinatives", []) or [])

    sel = _select_main_and_alts_from_scores(state)
    scored_main = sel["main_dets"]
    scored_alts = sel["alt_sets"]
    main_per_word = sel.get("main_per_word", []) or []
    alt_per_word = sel.get("alt_per_word", []) or []

    def _split_sem_fun(det_list: List[Dict[str, Any]]) -> tuple:
        sem=[]
        fun=[]
        for d in det_list or []:
            if not isinstance(d, dict):
                continue
            tok = (d.get("token") or d.get("sign") or "").strip()
            if tok in PLURAL_MARKERS:
                fun.append(d)
            else:
                sem.append(d)
        return sem, fun


    # Decide whether to override LLM main:
    # override if LLM main contains a hard-excluded sign, or if scored_main has higher coverage of strong signs
    llm_set = {(d["token"], int(d["index"])) for d in llm_main}
    scored_set = {(d["token"], int(d["index"])) for d in scored_main}

    def _set_strength(sset: set) -> int:
        # Count strong determinative tokens
        return sum(1 for (tok, _) in sset if tok in STRONG_DETERMINATIVE_SIGNS)

    override = False
    if any(d["token"] in NEVER_DETERMINATIVE_SIGNS for d in llm_main):
        override = True
    # If LLM main is tiny but we have richer scored_main with multi-source support, override
    if len(llm_set) <= 1 and len(scored_set) >= 2 and _set_strength(scored_set) >= _set_strength(llm_set) + 1:
        override = True
    # If LLM main misses plural markers Z* that are strongly supported
    if any(tok in {"Z1","Z2","Z3"} for (tok, _) in scored_set) and not any(tok in {"Z1","Z2","Z3"} for (tok, _) in llm_set):
        override = True

    if override:
        synth_data.setdefault("main_interpretation", {})
        synth_data["main_interpretation"]["determinatives"] = scored_main
        # Populate semantic_classifiers vs functional_markers for MAIN (determinatives remain union)
        sem_main, fun_main = _split_sem_fun(scored_main)
        synth_data["main_interpretation"]["semantic_classifiers"] = sem_main
        synth_data["main_interpretation"]["functional_markers"] = fun_main

        # Populate per_word spans (determinative blocks) to enforce suffix-span structure
        per_word_out = []
        if isinstance(main_per_word, list) and main_per_word:
            for wsel in main_per_word:
                if not isinstance(wsel, dict):
                    continue
                wid = int(wsel.get("word_id", 0))
                span_idx = wsel.get("span_token_indices") or []
                # determinatives actually selected for this word (by word_id)
                w_dets = [d for d in scored_main if isinstance(d, dict) and int(d.get("word_id", -1)) == wid]
                w_sem, w_fun = _split_sem_fun(w_dets)
                per_word_out.append({
                    "word_id": wid,
                    "span_token_indices": [int(i) for i in span_idx] if isinstance(span_idx, list) else [],
                    "text": wsel.get("text") or "",
                    "semantic_classifiers": w_sem,
                    "functional_markers": w_fun,
                    "determinatives": w_dets,
                })
        synth_data["main_interpretation"]["per_word"] = per_word_out

        # Append an explicit note to justification/validation to keep transparency
        just = (synth_data["main_interpretation"].get("justification") or "").strip()
        note = "Nota: el conjunto de determinativos fue ajustado por consolidación automática (semántica+estructura+reglas) para reducir falsos positivos."
        if note not in just:
            synth_data["main_interpretation"]["justification"] = (just + "\n" + note).strip() if just else note

        # Replace / merge alternatives: keep existing + add scored alternatives (dedup)
        existing_alts = synth_data.get("alternative_interpretations", []) or []
        merged: List[Dict[str, Any]] = []
        # Keep existing (filtered) first
        for alt in existing_alts:
            if not isinstance(alt, dict):
                continue
            dets = _filter_det_list(alt.get("determinatives", []) or [])
            if not dets:
                continue
            merged.append({
                "determinatives": dets,
                "perspective": (alt.get("perspective") or "Mixta"),
                "reason": (alt.get("reason") or "").strip(),
            })
        # Add scored alternatives
        for i, s in enumerate(scored_alts):
            sem_s, fun_s = _split_sem_fun(s)
            pw = alt_per_word[i] if i < len(alt_per_word) else []
            per_word_alt = []
            if isinstance(pw, list) and pw:
                for wsel in pw:
                    if not isinstance(wsel, dict):
                        continue
                    wid = int(wsel.get("word_id", 0))
                    span_idx = wsel.get("span_token_indices") or []
                    w_dets = [d for d in s if isinstance(d, dict) and int(d.get("word_id", -1)) == wid]
                    w_sem, w_fun = _split_sem_fun(w_dets)
                    per_word_alt.append({
                        "word_id": wid,
                        "span_token_indices": [int(j) for j in span_idx] if isinstance(span_idx, list) else [],
                        "text": wsel.get("text") or "",
                        "semantic_classifiers": w_sem,
                        "functional_markers": w_fun,
                        "determinatives": w_dets,
                    })
            merged.append({
                "semantic_classifiers": sem_s,
                "functional_markers": fun_s,
                "determinatives": s,
                "per_word": per_word_alt,
                "perspective": "Mixta",
                "reason": "Alternativa generada por consolidación automática de evidencias (selección por spans sufijo por palabra).",
            })

# Deduplicate by set
        seen = set()
        out_alts = []
        for alt in merged:
            sset = tuple(sorted((d["token"], int(d["index"])) for d in (alt.get("determinatives") or [])))
            if not sset or sset in seen or sset == tuple(sorted(scored_set)):
                continue
            seen.add(sset)
            out_alts.append(alt)
        synth_data["alternative_interpretations"] = out_alts

    else:
        # Always filter hard exclusions from existing
        synth_data.setdefault("main_interpretation", {})
        synth_data["main_interpretation"]["determinatives"] = llm_main
        sem_main, fun_main = _split_sem_fun(llm_main)
        synth_data["main_interpretation"]["semantic_classifiers"] = sem_main
        synth_data["main_interpretation"]["functional_markers"] = fun_main

        # Filter alternatives too
        alts = synth_data.get("alternative_interpretations", []) or []
        new_alts=[]
        for alt in alts:
            if not isinstance(alt, dict):
                continue
            dets = _filter_det_list(alt.get("determinatives", []) or [])
            if dets:
                alt2=dict(alt)
                alt2["determinatives"]=dets
                new_alts.append(alt2)
        synth_data["alternative_interpretations"]=new_alts

    return synth_data




def _format_det_list(det_list: List[Dict[str, Any]]) -> str:
    """Format determinatives as: [R16 (pos 24), W22VARA (pos 4)] or [] (ninguno incluido)."""
    if not isinstance(det_list, list) or len(det_list) == 0:
        return "[] (ninguno incluido)"
    parts = []
    for d in det_list:
        if not isinstance(d, dict):
            continue
        tok = (d.get("token") or d.get("sign") or "").strip()
        idx = d.get("index", d.get("token_index", d.get("position", None)))
        try:
            idx = int(idx)
        except Exception:
            idx = None
        if tok:
            parts.append(f"{tok} (pos {idx})" if idx is not None else tok)
    inner = ", ".join(parts) if parts else ""
    return f"[{inner}]" if inner else "[] (ninguno incluido)"


def format_report_from_synth_json(synth_data: Dict[str, Any]) -> str:
    """
    Build a human-readable report mirroring the previous structure, using ONLY the synthesizer JSON
    (JSON end-to-end: no regex parsing of free text).
    Output language: Spanish (section headers match legacy outputs).
    """
    if not isinstance(synth_data, dict):
        return ""

    main = synth_data.get("main_interpretation", {}) or {}
    main_det = main.get("determinatives", []) or []
    main_det_str = _format_det_list(main_det)

    justification = (main.get("justification") or "").strip()
    validation = (main.get("validation") or "").strip()

    alts = synth_data.get("alternative_interpretations", []) or []
    ambiguity = synth_data.get("ambiguity", {}) or {}
    recs = synth_data.get("recommendation", []) or []

    lines: List[str] = []

    # MAIN
    lines.append("INTERPRETACIÓN PRINCIPAL:")
    # Prefer explicit categories when available
    main_sem = main.get("semantic_classifiers", []) or []
    main_fun = main.get("functional_markers", []) or []
    if main_sem or main_fun:
        lines.append(f"- Clasificadores semánticos: {_format_det_list(main_sem)}")
        lines.append(f"- Marcadores funcionales: {_format_det_list(main_fun)}")
        lines.append(f"- Determinativos (total): {main_det_str}")
    else:
        lines.append(f"- Determinativos: {main_det_str}")
    if justification:
        lines.append("- Justificación:")
        for ln in justification.splitlines():
            lines.append(f"  {ln}".rstrip())
    if validation:
        lines.append("- Validación:")
        for ln in validation.splitlines():
            lines.append(f"  {ln}".rstrip())

    lines.append("")  # spacer

    # ALTERNATIVES
    lines.append("INTERPRETACIONES ALTERNATIVAS:")
    if isinstance(alts, list) and len(alts) > 0:
        for i, alt in enumerate(alts, start=1):
            if not isinstance(alt, dict):
                continue
            alt_sem = alt.get("semantic_classifiers", []) or []
            alt_fun = alt.get("functional_markers", []) or []
            alt_det = alt.get("determinatives", []) or []
            alt_det_str = _format_det_list(alt_det).replace(" (ninguno incluido)", "")
            alt_sem_str = _format_det_list(alt_sem).replace(" (ninguno incluido)", "")
            alt_fun_str = _format_det_list(alt_fun).replace(" (ninguno incluido)", "")
            perspective = (alt.get("perspective") or "").strip()
            reason = (alt.get("reason") or "").strip()

            lines.append(f"- Alternativa {i}:")
            if alt_sem or alt_fun:
                lines.append(f"  • Clasificadores semánticos: {alt_sem_str if alt_sem_str else '[]'}")
                lines.append(f"  • Marcadores funcionales: {alt_fun_str if alt_fun_str else '[]'}")
                lines.append(f"  • Determinativos (total): {alt_det_str if alt_det_str else '[]'}")
            else:
                lines.append(f"  • Signos: {alt_det_str if alt_det_str else '[]'}")
            if perspective:
                lines.append(f"  • Perspectiva: {perspective}")
            if reason:
                lines.append(f"  • Razón para alternativa: {reason}")
    else:
        lines.append("- Ninguna.")

    lines.append("")

    # AMBIGUITY
    lines.append("AMBIGÜEDAD:")
    factors = ambiguity.get("factors", []) if isinstance(ambiguity, dict) else []
    consensus = (ambiguity.get("consensus") or "").strip() if isinstance(ambiguity, dict) else ""
    if isinstance(factors, list) and len(factors) > 0:
        lines.append("- Factores:")
        for fct in factors:
            if fct:
                lines.append(f"  • {str(fct).strip()}")
    else:
        lines.append("- Factores: (no especificados)")
    if consensus:
        lines.append("- Consenso:")
        for ln in consensus.splitlines():
            lines.append(f"  {ln}".rstrip())

    lines.append("")

    # RECOMMENDATION
    lines.append("RECOMENDACIÓN:")
    if isinstance(recs, list) and len(recs) > 0:
        for r in recs:
            if r:
                lines.append(f"- {str(r).strip()}")
    else:
        rec_text = (synth_data.get("recommendation_text") or "").strip()
        if rec_text:
            for ln in rec_text.splitlines():
                lines.append(f"- {ln}".rstrip())
        else:
            lines.append("- (sin recomendación explícita)")

    # End block separator for consistency with legacy formatting
    lines.append("")
    lines.append("═══════════════════════════════")

    return "\n".join(lines)

# ============================================================================
# REGRAS DO ESPECIALISTA - V1.5 (Baseado em Feedback Real)
# ============================================================================

# EXCLUDED_SIGNS: Signos que RARAMENTE/NUNCA são determinativos
# Baseado em análise de 50 casos com correções de especialista
# Taxa de erro identificada: 46.3% → Esperado após correção: ~15-20%

RARELY_DETERMINATIVES = {
    'G17': {
        'reason': 'Praticamente NUNCA é determinativo',
        'function': 'Fonograma muito comum (som /m/)',
        'error_frequency': 8,  # Identificado incorretamente 8x
        'confidence_penalty': 0.1  # Reduz confiança para 10%
    },
    'N35': {
        'reason': 'NUNCA é determinativo sozinho',
        'function': 'Fonograma muito comum (som /n/)',
        'exception': 'Só como N35A (3x N35 sequenciais) = determinativo de água',
        'error_frequency': 20,  # Erro mais frequente!
        'confidence_penalty': 0.05  # Reduz confiança para 5%
    },
    'S29': {
        'reason': 'Muito IMPROVÁVEL ser determinativo',
        'function': 'Fonograma comum (som /s/)',
        'error_frequency': 10,
        'confidence_penalty': 0.15
    },
    'M17': {
        'reason': 'Raramente é determinativo',
        'function': 'Fonograma muito comum (som /i/ ou /y/)',
        'error_frequency': 15,  # Segundo erro mais frequente!
        'confidence_penalty': 0.2
    },
    'D21': {
        'reason': 'Raramente é determinativo',
        'function': 'Fonograma comum (som /r/ - boca)',
        'error_frequency': 5,
        'confidence_penalty': 0.25
    },
    'D2': {
        'reason': 'Geralmente fonograma ou ideograma',
        'function': 'Rosto - fonograma',
        'error_frequency': 5,
        'confidence_penalty': 0.25
    }
}

# Signos COMUNS e CONFIÁVEIS (baseado em acertos do especialista)
COMMON_DETERMINATIVES = {
    'A1': 20,   # Homem sentado - 20 acertos
    'Z1': 20,   # Traço vertical (plural) - 20 acertos
    'Z2': 11,   # Três traços (plural) - 11 acertos
    'D54': 10,  # Pernas caminhando - 10 acertos
    'N1': 8,    # Céu - 8 acertos
    'A40': 8,   # Deus sentado - 8 acertos
    'B1': 5,    # Mulher sentada - 5 acertos
}

# ============================================================================
# V1.6: PRIORITY_SIGNS E MELHORIAS DE RECALL (Falsos Negativos)
# ============================================================================

# PRIORITY_SIGNS: Signos que SEMPRE devem ser incluídos se presentes
# Baseado em análise de 48 falsos negativos
# Problema identificado: Taxa de recall de apenas 67% (deveria ser 85-90%)

ALWAYS_INCLUDE = {
    # MARCADORES DE PLURAL (13+3+1 = 17 perdas!)
    'Z1': {
        'reason': 'MARCADOR DE PLURAL - sempre determinativo',
        'priority': 'CRITICAL',  # Perdido 13x!
        'confidence': 0.95,
        'type': 'plural_marker'
    },
    'Z2': {
        'reason': 'MARCADOR DE PLURAL - sempre determinativo',
        'priority': 'CRITICAL',  # Perdido 3x
        'confidence': 0.95,
        'type': 'plural_marker'
    },
    'Z3': {
        'reason': 'MARCADOR DE PLURAL - sempre determinativo',
        'priority': 'HIGH',
        'confidence': 0.95,
        'type': 'plural_marker'
    },
    
    # DETERMINATIVOS MUITO COMUNS (nunca deveriam ser perdidos)
    'A1': {
        'reason': 'Homem sentado - determinativo muito comum',
        'priority': 'HIGH',
        'confidence': 0.9,
        'type': 'person'
    },
    'B1': {
        'reason': 'Mulher sentada - determinativo muito comum',
        'priority': 'HIGH',
        'confidence': 0.9,
        'type': 'person'
    },
    'A40': {
        'reason': 'Deus sentado - determinativo muito comum',
        'priority': 'HIGH',
        'confidence': 0.9,
        'type': 'divine'
    },
    'D54': {
        'reason': 'Pernas caminhando - determinativo muito comum',
        'priority': 'HIGH',
        'confidence': 0.9,
        'type': 'action'
    }
}


def identify_plural_markers(tokens: List[str]) -> List[Dict]:
    """
    Identifica marcadores de plural com ALTA confiança
    
    V1.6: Z1, Z2, Z3 são SEMPRE determinativos quando em contexto de plural
    Problema identificado: Z1 foi perdido 13x!
    
    Args:
        tokens: Lista de tokens da sequência
    
    Returns:
        Lista de marcadores de plural identificados
    """
    plural_markers = []
    
    for i, token in enumerate(tokens):
        if token in ['Z1', 'Z2', 'Z3']:
            # Z1/Z2/Z3 após qualquer token = provavelmente plural
            if i > 0:  # Não é o primeiro token
                plural_markers.append({
                    'sign': token,
                    'position': i,
                    'confidence': 0.95,  # Confiança muito alta!
                    'type': 'plural_marker',
                    'method': 'plural_detection_v16',
                    'context': 'after_word',
                    'reason': 'Marcador de plural detectado'
                })
    
    return plural_markers


def analyze_sequential_patterns(tokens: List[str]) -> List[Dict]:
    """
    Detecta padrões sequenciais comuns que indicam determinativos
    
    V1.6: Analisa PADRÕES, não só signos isolados
    
    Padrões detectados:
    - DET-Z1: determinativo + plural
    - DET1-DET2: duplo determinativo
    
    Args:
        tokens: Lista de tokens da sequência
    
    Returns:
        Lista de determinativos detectados por padrão
    """
    pattern_dets = []
    
    for i in range(len(tokens) - 1):
        current = tokens[i]
        next_token = tokens[i + 1]
        
        # Padrão 1: DET-Z1/Z2/Z3 (determinativo + plural)
        if current in DETERMINATIVES_DETAILED and next_token in ['Z1', 'Z2', 'Z3']:
            # O plural sempre deve ser incluído
            if next_token not in [d['sign'] for d in pattern_dets]:
                pattern_dets.append({
                    'sign': next_token,
                    'position': i + 1,
                    'confidence': 0.95,
                    'pattern': 'DET-PLURAL',
                    'method': 'pattern_detection_v16',
                    'reason': f'Plural após {current}'
                })
        
        # Padrão 2: DET1-DET2 (duplo determinativo)
        if (current in DETERMINATIVES_DETAILED and 
            next_token in DETERMINATIVES_DETAILED and
            next_token not in RARELY_DETERMINATIVES):  # Não se está na excluded_signs
            
            # Bonificar o segundo determinativo
            if next_token not in [d['sign'] for d in pattern_dets]:
                pattern_dets.append({
                    'sign': next_token,
                    'position': i + 1,
                    'confidence': 0.85,
                    'pattern': 'DOUBLE-DET',
                    'method': 'pattern_detection_v16',
                    'reason': f'Duplo determinativo após {current}'
                })
    
    return pattern_dets


def force_priority_signs_determinatives(predicted_dets: List[Dict], tokens: List[str]) -> List[Dict]:
    """
    FORÇA inclusão de signos da priority_signs que estão presentes mas não foram detectados
    
    V1.6: Garante que Z1, Z2, Z3, A1, B1, A40, D54 sejam SEMPRE incluídos
    
    Args:
        predicted_dets: Determinativos já identificados
        tokens: Todos os tokens da sequência
    
    Returns:
        Lista expandida com determinativos forçados
    """
    expanded = predicted_dets.copy()
    detected_signs = {d['sign'] for d in expanded}
    
    print("\n🔍 V1.6: Verificando priority_signs...")
    
    forced_count = 0
    for token in tokens:
        if token in ALWAYS_INCLUDE and token not in detected_signs:
            # Este signo DEVERIA ter sido detectado!
            info = ALWAYS_INCLUDE[token]
            
            # Verificar se está na base de conhecimento
            if token in DETERMINATIVES_DETAILED:
                det_info = DETERMINATIVES_DETAILED[token]
                
                expanded.append({
                    'sign': token,
                    'position': tokens.index(token),
                    'significado': det_info['significado'],
                    'categoria': det_info['categoria'],
                    'semantica': det_info['semantica'],
                    'confidence': info['confidence'],
                    'method': 'priority_signs_forced_v16',
                    'priority': info['priority'],
                    'reason': info['reason']
                })
                
                forced_count += 1
                print(f"  ➕ {token} FORÇADO (priority_signs): {info['reason']}")
    
    if forced_count > 0:
        print(f"  ✅ {forced_count} determinativos adicionados pela priority_signs")
    else:
        print("  ✓ Todos os signos da priority_signs já detectados")
    
    return expanded


def is_likely_phonogram(sign: str, sequence_context: str = "") -> bool:
    """
    Verifica se um signo é provavelmente um fonograma (NÃO determinativo)
    Baseado em feedback do especialista
    
    Args:
        sign: Código do signo (ex: 'G17', 'N35')
        sequence_context: Sequência completa para contexto
    
    Returns:
        True se é provavelmente fonograma (deve ser excluído)
    """
    # Verificar excluded_signs
    if sign in RARELY_DETERMINATIVES:
        # Regra especial para N35
        if sign == 'N35':
            # Verificar se é N35A (3x N35 sequenciais)
            if 'N35-N35-N35' in sequence_context or 'N35:N35:N35' in sequence_context:
                return False  # É N35A, pode ser determinativo
            else:
                return True  # N35 sozinho = SEMPRE fonograma
        
        # Outros signos da excluded_signs são muito provavelmente fonogramas
        return True
    
    return False


def calculate_confidence_v15(sign: str, position: str, context: str = "") -> float:
    """
    Calcula confiança de um determinativo considerando feedback do especialista
    
    V1.5: Penaliza signos problemáticos, bonifica signos confiáveis
    
    Args:
        sign: Código do signo
        position: Posição na sequência
        context: Contexto adicional
    
    Returns:
        Confiança de 0.0 a 1.0
    """
    # Confiança base (posição final = boa)
    base_confidence = 0.7 if position == 'final' else 0.5
    
    # PENALIZAR signos problemáticos (excluded_signs)
    if sign in RARELY_DETERMINATIVES:
        penalty = RARELY_DETERMINATIVES[sign]['confidence_penalty']
        base_confidence *= penalty
        
        # Se confiança cai abaixo de 0.3, provavelmente é falso positivo
        if base_confidence < 0.3:
            return 0.1  # Confiança muito baixa
    
    # BONIFICAR signos confiáveis (lista de acertos)
    if sign in COMMON_DETERMINATIVES:
        frequency = COMMON_DETERMINATIVES[sign]
        # Quanto mais frequente nos acertos, maior bônus
        bonus = min(1.5, 1.0 + (frequency / 50))
        base_confidence *= bonus
    
    # Limitar entre 0.0 e 1.0
    return min(1.0, max(0.0, base_confidence))


def validate_determinatives_v15(predicted_dets: List[Dict], sequence: str = "") -> List[Dict]:
    """
    Valida determinativos identificados usando regras do especialista
    
    V1.5: Filtra falsos positivos baseado em feedback real
    
    Args:
        predicted_dets: Lista de determinativos preditos
        sequence: Sequência MdC completa para contexto
    
    Returns:
        Lista filtrada de determinativos validados
    """
    validated = []
    
    for det in predicted_dets:
        sign = det.get('sign', '')
        
        # Regra 1: Verificar se é provavelmente fonograma
        if is_likely_phonogram(sign, sequence):
            print(f"  ⚠️  {sign} DESCARTADO: {RARELY_DETERMINATIVES[sign]['reason']}")
            continue
        
        # Regra 2: Verificar confiança mínima
        confidence = calculate_confidence_v15(
            sign, 
            det.get('position', 'unknown'),
            sequence
        )
        
        # Threshold: só aceitar se confiança >= 0.3
        if confidence < 0.3:
            print(f"  ⚠️  {sign} DESCARTADO: Confiança muito baixa ({confidence:.2f})")
            continue
        
        # Atualizar confiança no determinativo
        det['confidence'] = confidence
        det['validated_v15'] = True
        
        validated.append(det)
    
    return validated


def validate_determinatives_v16(predicted_dets: List[Dict], tokens: List[str], sequence: str = "") -> List[Dict]:
    """
    V1.6: Validação COMPLETA - Filtra falsos positivos E aumenta recall
    
    Combina:
    - V1.5: Filtragem de falsos positivos (excluded_signs)
    - V1.6: Aumento de recall (priority_signs + padrões + plurais)
    
    Args:
        predicted_dets: Determinativos pelos agentes
        tokens: Lista completa de tokens da sequência
        sequence: Sequência MdC completa para contexto
    
    Returns:
        Lista validada e expandida de determinativos
    """
    print("\n🔍 V1.6: VALIDAÇÃO COMPLETA (Precisão + Recall)...")
    print("="*70)
    
    # ========================================================================
    # ETAPA 1: FILTRAR FALSOS POSITIVOS (V1.5)
    # ========================================================================
    print("\n📍 Etapa 1/4: Filtrando falsos positivos (excluded_signs)...")
    
    validated = []
    for det in predicted_dets:
        sign = det.get('sign', '')
        
        # Verificar excluded_signs
        if is_likely_phonogram(sign, sequence):
            print(f"  ⚠️  {sign} DESCARTADO: {RARELY_DETERMINATIVES[sign]['reason']}")
            continue
        
        # Verificar confiança mínima
        confidence = calculate_confidence_v15(
            sign, 
            det.get('position', 'unknown'),
            sequence
        )
        
        if confidence < 0.3:
            print(f"  ⚠️  {sign} DESCARTADO: Confiança baixa ({confidence:.2f})")
            continue
        
        det['confidence'] = confidence
        det['validated_v16'] = True
        validated.append(det)
    
    print(f"  ✅ Etapa 1 completa: {len(validated)} determinativos validados")
    
    # ========================================================================
    # ETAPA 2: IDENTIFICAR MARCADORES DE PLURAL (V1.6)
    # ========================================================================
    print("\n📍 Etapa 2/4: Identificando marcadores de plural (Z1, Z2, Z3)...")
    
    plural_markers = identify_plural_markers(tokens)
    
    # Adicionar plurais que não foram detectados
    detected_signs = {d['sign'] for d in validated}
    for marker in plural_markers:
        if marker['sign'] not in detected_signs:
            # Completar info da base de conhecimento
            if marker['sign'] in DETERMINATIVES_DETAILED:
                info = DETERMINATIVES_DETAILED[marker['sign']]
                marker.update({
                    'significado': info['significado'],
                    'categoria': info['categoria'],
                    'semantica': info['semantica']
                })
            
            validated.append(marker)
            print(f"  ➕ {marker['sign']} ADICIONADO: {marker['reason']}")
    
    print(f"  ✅ Etapa 2 completa: +{len(plural_markers)} marcadores de plural")
    
    # ========================================================================
    # ETAPA 3: ANALISAR PADRÕES SEQUENCIAIS (V1.6)
    # ========================================================================
    print("\n📍 Etapa 3/4: Analisando padrões sequenciais...")
    
    pattern_dets = analyze_sequential_patterns(tokens)
    
    # Adicionar determinativos detectados por padrões
    detected_signs = {d['sign'] for d in validated}
    for det in pattern_dets:
        if det['sign'] not in detected_signs:
            # Completar info da base
            if det['sign'] in DETERMINATIVES_DETAILED:
                info = DETERMINATIVES_DETAILED[det['sign']]
                det.update({
                    'significado': info['significado'],
                    'categoria': info['categoria'],
                    'semantica': info['semantica']
                })
            
            validated.append(det)
            print(f"  ➕ {det['sign']} ADICIONADO: Padrão {det['pattern']}")
    
    print(f"  ✅ Etapa 3 completa: +{len(pattern_dets)} por padrões")
    
    # ========================================================================
    # ETAPA 4: FORÇAR PRIORITY_SIGNS (V1.6)
    # ========================================================================
    print("\n📍 Etapa 4/4: Verificando priority_signs (ALWAYS_INCLUDE)...")
    
    validated = force_priority_signs_determinatives(validated, tokens)
    
    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    print("\n" + "="*70)
    print("📊 VALIDAÇÃO V1.6 COMPLETA:")
    print(f"   Entrada: {len(predicted_dets)} determinativos preditos")
    print(f"   Saída: {len(validated)} determinativos validados")
    print(f"   Diferença: {len(validated) - len(predicted_dets):+d}")
    print("="*70)
    
    return validated


# Base de conocimiento com determinativos de Kammerzell 2024

# ============================================================================
# 🎯 REGRAS GLOBAIS DE ANÁLISE DE DETERMINATIVOS
# ============================================================================
# Estas regras aplicam-se a TODOS os agentes para evitar redundâncias
# e garantir consistência na análise.
# ============================================================================

GLOBAL_DETERMINATIVE_RULES = """
═══════════════════════════════════════════════════════════════════════
REGRAS GLOBAIS - APLICAM-SE A TODOS OS AGENTES
═══════════════════════════════════════════════════════════════════════

1. EXCLUSÕES OBRIGATÓRIAS (salvo contexto excepcional):
   • S29: hífen (-), NUNCA é determinativo
   • N35: água simples, geralmente fonograma, raramente determinativo
   
2. POLIVALENTES (requerem evidência FORTE para inclusão):
   • M17: fonograma /i/ ou /y/ (só determinativo se contexto claro)
   • D21: boca/r (só determinativo se palavra sobre fala/comunicação)
   • Aa2: pústula (requer evidência semântica clara)
   
3. PLURAIS (SEMPRE incluir quando identificados):
   • Z1, Z2, Z3: Liste CADA ocorrência separadamente
   
4. ESCALA DE CONFIANÇA GLOBAL:
   • 0.90-1.00: ALTA | 0.75-0.89: MÉDIA-ALTA | 0.60-0.74: MÉDIA
   
5. CASO SEM DETERMINATIVOS:
   • É VÁLIDO que sequências NÃO tenham determinativos
   • Se não há evidência clara → retornar []
   
6. REGRA CRÍTICA: REPETIÇÕES (ABSOLUTAMENTE FUNDAMENTAL):
   ═══════════════════════════════════════════════════════════════════
   Se um determinativo aparece MÚLTIPLAS VEZES:
   Liste CADA ocorrência SEPARADAMENTE
   EXEMPLO: Aa2 3x → Resposta: [Aa2, Aa2, Aa2]
   NUNCA: [Aa2] ou "Aa2 (3x)" ou "padrão consolidado"
   ═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════
"""


DETERMINATIVES_DETAILED = {
    # PERSONA Y SOCIEDAD
    'B1': {'significado': 'persona femenina', 'categoria': 'person', 'semantica': 'mujer, hermana, hija, sacerdotisa, sirvienta'},
    'A1': {'significado': 'persona masculina', 'categoria': 'person', 'semantica': 'hombre, hermano, hijo, sacerdote, sirviente'},
    'A2': {'significado': 'hombre con mano en la boca', 'categoria': 'action', 'semantica': 'comer, beber, hablar, pensar, sentir'},
    'A40': {'significado': 'hombre con barba ceremonial', 'categoria': 'divine', 'semantica': 'dios, Horus, dios solar, señor'},
    'A17': {'significado': 'niño', 'categoria': 'person', 'semantica': 'niño, huérfano, joven, pequeño'},
    'A53': {'significado': 'momia erguida', 'categoria': 'abstract', 'semantica': 'momia, estatua, figura, forma, ser, semejarse'},
    
    # PARTES DEL CUERPO
    'D4': {'significado': 'ojo humano', 'categoria': 'body', 'semantica': 'ojo, ver, contemplar, soñar, vigilar'},
    'D19': {'significado': 'ojo, nariz y mejilla', 'categoria': 'body', 'semantica': 'nariz, rostro, oler, alegría, amigable'},
    'D40': {'significado': 'brazo con bastón', 'categoria': 'action', 'semantica': 'golpear, fuerte, fuerza, victorioso, trabajar, fabricar, artesano, hábil, tomar, capturar, robar, tocar, ocultar'},
    'D54': {'significado': 'pantorrilla y pies', 'categoria': 'action',
            'semantica': 'ir, caminar, viajar, entrar, carrera, procesión',
            'disambiguation': 'ONLY determinative in motion/travel contexts (ir, caminar, venir, entrar). '
                              'Requires explicit movement verb or travel concept in the sequence. '
                              'NOT a determinative for general human figures, work actions, or military contexts. '
                              'When paired with A1/A24: the human figure (A1/A24) classifies the person, '
                              'D54 classifies the motion — both can coexist.'},
    'D3': {'significado': 'mechones de pelo', 'categoria': 'body', 'semantica': 'pelo, cabello, peluquero, peluca, viuda, luto, lamento, calvo'},
    'Aa2': {'significado': 'pústula o glándula (?)', 'categoria': 'abstract', 'semantica': 'embalsamador, descomponerse, herida, dañino, malo'},
    
    # ANIMALES
    'E1': {'significado': 'ganado vacuno', 'categoria': 'animal', 'semantica': 'ganado vacuno, toro, vaca, ganado mayor'},
    'E8': {'significado': 'cabrito', 'categoria': 'animal', 'semantica': 'cabrito, ganado menor'},
    'F27': {'significado': 'piel', 'categoria': 'body', 'semantica': 'piel, cuero, piel animal, correa, taparrabos, ratón, gato'},
    'G7': {'significado': 'halcón sobre estandarte', 'categoria': 'divine', 'semantica': 'dios, rey'},
    'G39': {'significado': 'pato', 'categoria': 'animal', 'semantica': 'pato, ave, garza, langosta',
            'disambiguation': 'EASILY CONFUSED with H8 (duck egg), A47 (shepherd with staff). '
                              'G39 = swallow/duck bird, often a logogram not determinative. '
                              'Only accept as determinative in explicit bird/waterfowl context.'},
    'G40': {'significado': 'pato cuchara volando', 'categoria': 'action', 'semantica': 'volar'},
    'G37': {'significado': 'gorrión', 'categoria': 'abstract', 'semantica': 'pequeño, más joven, pobre, malo, enfermo, perecer, maldad'},
    'I14': {'significado': 'serpiente', 'categoria': 'animal', 'semantica': 'serpiente, gusano'},
    'K1': {'significado': 'pez', 'categoria': 'animal', 'semantica': 'pez'},
    
    # NATURALEZA Y PLANTAS
    'M1': {'significado': 'árbol', 'categoria': 'nature', 'semantica': 'sicómoro, espino'},
    'M2': {'significado': 'junco', 'categoria': 'nature', 'semantica': 'papiro, planta, flor, raíz'},
    'M3': {'significado': 'rama', 'categoria': 'nature', 'semantica': 'ébano, caja, viga, mástil'},
    'M33': {'significado': 'granos de cereal', 'categoria': 'nature', 'semantica': 'cereal, cebada, grano'},
    
    # CIELO Y COSMOS
    'N5': {'significado': 'disco solar', 'categoria': 'sky', 'semantica': 'sol, salir, día, hora, tiempo'},
    'N25': {'significado': 'colina', 'categoria': 'land', 'semantica': 'extranjero, desierto, montaña'},
    'N33': {'significado': 'gránulo', 'categoria': 'material', 'semantica': 'arena, perla, mineral, gránulo, píldora'},
    'N34': {'significado': 'barra de metal', 'categoria': 'material', 'semantica': 'bronce, hierro, espejo, daga'},
    'N35A': {'significado': 'líneas de agua', 'categoria': 'nature', 'semantica': 'agua, líquido, lavar, beber'},
    'N36': {'significado': 'sección de canal', 'categoria': 'nature', 'semantica': 'río, lago, canal'},
    
    # EDIFICIOS Y CONSTRUCCIONES
    'O1': {'significado': 'plano de casa', 'categoria': 'building', 'semantica': 'casa, edificio, localidad, dominio, estate',
           'disambiguation': 'POSITION-DEPENDENT (v1.38 fix): '
                             'O1 at INDEX 0-1 = LOGOGRAM (it IS the word house/estate as subject), REJECT as determinative. '
                             'O1 at INDEX >= 2 (middle or end) = POSSIBLE DETERMINATIVE for building/estate/domain words. '
                             'When O1 appears after phonetic signs mid-sequence, it may classify the word domain/temple/estate.'},
    
    # ABSTRACTOS Y ESCRITURA
    'Y1': {'significado': 'rollo de papiro', 'categoria': 'abstract', 'semantica': 'rollo de libro, documento escrito, decreto, suma, número, palabra, cosa, hechizo, magia, saber, poder'},
    
    # PLURALES Y MARCADORES
    'Z2': {'significado': 'tres trazos', 'categoria': 'grammar', 'semantica': 'pluralidad'},
    
    # LOGOGRAMAS (Sección 2-8 del documento)
    'A12': {'significado': 'soldado con arco', 'categoria': 'person', 'semantica': 'ejército, tropas'},
    'E17': {'significado': 'chacal', 'categoria': 'animal', 'semantica': 'chacal'},
    'F4': {'significado': 'cabeza de toro', 'categoria': 'body', 'semantica': 'parte delantera, frente, príncipe, cuerda de proa'},
    'F21': {'significado': 'oreja', 'categoria': 'body', 'semantica': 'oír, obedecer'},
    'L2': {'significado': 'abeja', 'categoria': 'animal', 'semantica': 'abeja, miel, rey de Bajo Egipto'},
    'L1': {'significado': 'escarabajo', 'categoria': 'animal', 'semantica': 'llegar a ser, surgir, transformarse'},
    'N1': {'significado': 'cielo', 'categoria': 'sky', 'semantica': 'cielo'},
    'N11': {'significado': 'luna creciente', 'categoria': 'sky', 'semantica': 'luna'},
    'N14': {'significado': 'estrella', 'categoria': 'sky', 'semantica': 'estrella, alabar, enseñar'},
    'O42': {'significado': 'cercado rectangular', 'categoria': 'action', 'semantica': 'agarrar, asir'},
    'R8': {'significado': 'bandera sobre bastón (dios)', 'categoria': 'divine', 'semantica': 'dios, diosa, alabar a dios, agradecer'},
    'S34': {'significado': 'ankh (vida)', 'categoria': 'abstract', 'semantica': 'vivir, vida, los vivientes, juramento, espejo'},
    'T18': {'significado': 'cayado', 'categoria': 'object', 'semantica': 'seguir, partidario, acompañante',
            'disambiguation': 'CONTEXT-DEPENDENT. Only determinative for follower/adherent/companion words. '
                              'Often a logogram not a determinative. REJECT in general human activity contexts.'},

    # V1.6: SIGNOS ADICIONADOS (baseado em análise de falsos negativos)
    'P5B': {'significado': 'barco (variante)', 'categoria': 'transport', 'semantica': 'navegación, viaje por agua, barca'},  # Perdido 2x
    'V1': {'significado': 'lazo', 'categoria': 'object', 'semantica': 'lazo, unir, atar, cien mil',
           'disambiguation': 'HIGHLY AMBIGUOUS — default = phonogram/numeral (100,000). '
                             'Only accept as determinative in explicit binding/tying contexts. '
                             'REJECT unless strong binding context is present.'},  # Perdido 2x
    'A193': {'significado': 'persona (variante)', 'categoria': 'person', 'semantica': 'persona, agente, variante de A1'},
    'O36': {'significado': 'muro alto con abertura', 'categoria': 'building', 'semantica': 'fortaleza, castillo, muro'},
    'A7A': {'significado': 'persona sentada (variante)', 'categoria': 'person', 'semantica': 'persona, variante'},
    'A24': {'significado': 'hombre con brazo extendido', 'categoria': 'person', 'semantica': 'acción con el brazo, golpear'},
    'A35': {'significado': 'hombre construyendo', 'categoria': 'person', 'semantica': 'constructor, trabajador, construir'},
    'D20': {'significado': 'nariz', 'categoria': 'body', 'semantica': 'nariz, oler, respirar, alegría'},
    'W24': {'significado': 'taza', 'categoria': 'object', 'semantica': 'vaso, recipiente, ofrecer'},
    'X4': {'significado': 'pan', 'categoria': 'food', 'semantica': 'pan, ofrenda'},
    'Y2': {'significado': 'papiro enrollado', 'categoria': 'abstract', 'semantica': 'libro, documento, palabra escrita'},
    'Q5': {'significado': 'sarcófago', 'categoria': 'funeral', 'semantica': 'sarcófago, ataúd, muerte'},

    # V1.37: SIGNOS ADICIONADOS (análise de FN_NULL — sinais ausentes da KB)
    'Aa3':  {'significado': 'rosto humano (variante)', 'categoria': 'body',
             'semantica': 'rosto, face, expressão, presença'},
    'E21':  {'significado': 'cão/chacal deitado', 'categoria': 'animal',
             'semantica': 'cão, chacal, Anúbis, animal doméstico, guarda',
             'disambiguation': 'V1.38: REQUIRES explicit canine or Anubis-worship context. '
                               'Only accept when sequence is specifically about dogs, jackals, or Anubis. '
                               'Do NOT predict as a generic animal classifier for non-canine contexts.'},
    'N31':  {'significado': 'estrada com árvores', 'categoria': 'land',
             'semantica': 'estrada, caminho, viajar, percurso, deserto'},
    'X3':   {'significado': 'bolo de pão redondo', 'categoria': 'food',
             'semantica': 'pão, alimento, ofrenda, comida',
             'disambiguation': 'V1.38: Only determinative in explicit offering lists with multiple food items. '
                               'REJECT in general ritual sequences where bread is implied but not the main classifier.'},
    'X6':   {'significado': 'grande pão redondo', 'categoria': 'food',
             'semantica': 'pão, ofrenda, alimento ritual',
             'disambiguation': 'V1.38: Same as X3 — only in explicit offering lists. '
                               'REJECT in sequences about general rituals or activities.'},
    
    # ========================================================================
    # V1.27: EXPANSÃO SISTEMÁTICA DA BASE DE CONHECIMENTO (+70 sinais)
    # Baseado em análise de FN e categorias completas de Gardiner
    # ========================================================================
    
    # SÉRIE F: PARTES DO CORPO (faltantes)
    'F51': {'significado': 'pedaço de carne com sangue', 'categoria': 'body', 
            'semantica': 'carne, sangue, açougueiro, sacrifício, corpo'},
    
    # SÉRIE N: PAISAGEM/ÁGUA (faltantes)
    'N16': {'significado': 'terreno cultivado', 'categoria': 'land',
            'semantica': 'terra, campo, parcela, agricultura',
            'disambiguation': 'SIMILAR TO N24 (pool/marshland). '
                              'N16 = dry cultivated field. N24 = pool/canal with water. '
                              'Do NOT predict N16 when the word is about water or pools.'},
    'N23': {'significado': 'canal com água', 'categoria': 'land', 
            'semantica': 'canal, irrigação, água, rego'},
    
    # SÉRIE T: FERRAMENTAS/ARMAS (faltantes)
    'T14': {'significado': 'machado de guerra', 'categoria': 'tool', 
            'semantica': 'machado, arma, guerra, combate, cortar'},
    
    # SÉRIE P: BARCOS (faltantes)
    'P1': {'significado': 'barco', 'categoria': 'transport', 
           'semantica': 'barco, embarcação, navegar, rio, viagem'},
    
    # SÉRIE O: EDIFÍCIOS (expandida)
    'O38': {'significado': 'canto de edifício', 'categoria': 'building', 
            'semantica': 'canto, ângulo, edifício, construção'},
    
    # SÉRIE D: PARTES DO CORPO (faltantes)
    'D56': {'significado': 'perna e pé', 'categoria': 'action', 
            'semantica': 'perna, caminhar, ir, mover-se'},
    
    # SÉRIE Z: TRAÇOS E MARCADORES (variantes)
    'Z6': {'significado': 'três traços (variante)', 'categoria': 'grammar', 
           'semantica': 'plural, múltiplo'},
    
}      
      
# Crear índices para búsqueda eficiente
DETERMINATIVES_BY_CATEGORY = defaultdict(list)
for sign, info in DETERMINATIVES_DETAILED.items():
    DETERMINATIVES_BY_CATEGORY[info['categoria']].append(sign)

DETERMINATIVES_BY_KEYWORD = defaultdict(list)
for sign, info in DETERMINATIVES_DETAILED.items():
    keywords = info['significado'].lower().split(', ')
    for keyword in keywords:
        DETERMINATIVES_BY_KEYWORD[keyword].append(sign)

print(f"✅ Base de conocimiento cargada: {len(DETERMINATIVES_DETAILED)} determinativos")
print(f"   Categorías: {len(DETERMINATIVES_BY_CATEGORY)}")
print(f"   Palabras clave: {len(DETERMINATIVES_BY_KEYWORD)}\n")

# ============================================================================
# FASE 3 - FUNCIONES DE ANÁLISE POR REGRAS
# ============================================================================

def analyze_rule_based(tokens):
    """
    Análise simples por regras usando a base de conhecimento
    """
    determinatives = []
    

    for i, token in enumerate(tokens):
        # Pular separadores
        if token in MDL_SEPARATORS:
            continue
            
        # Verificar se está na base de conhecimento
        if token in DETERMINATIVES_DETAILED:
            info = DETERMINATIVES_DETAILED[token]
            
            # Regra simples: determinativos geralmente aparecem no final
            is_final = (i == len(tokens) - 1) or (i == len(tokens) - 2 and tokens[-1] in MDL_SEPARATORS)
            
            determinatives.append({
                'sign': token,
                'position': i,
                'significado': info['significado'],
                'categoria': info['categoria'],
                'semantica': info['semantica'],
                'confidence': 'high' if is_final else 'medium',
                'method': 'rules'
            })
    
    return {
        'determinatives': determinatives,
        'count': len(determinatives),
        'method': 'rule_based'
    }

# ============================================================================
# FASE 4 - SISTEMA MULTI-AGENTE COM LANGGRAPH
# ============================================================================

# Contexto egiptológico (mesmo do CrewAI)
DETERMINATIVE_CONTEXT = """
CONOCIMIENTO EGIPTOLÓGICO SOBRE DETERMINATIVOS
================================================

Los determinativos son signos jeroglíficos que aparecen al final de las palabras
egipcias y no se leen fonéticamente. Su función es:

1. DESAMBIGUACIÓN: Indicar el significado correcto de palabras homófonas
2. CATEGORIZACIÓN SEMÁNTICA: Clasificar el concepto (persona, animal, acción, etc.)
3. ACLARACIÓN: Hacer el significado más preciso

CARACTERÍSTICAS DE LOS DETERMINATIVOS:
--------------------------------------

POSICIÓN:
- Generalmente aparecen AL FINAL de la palabra
- Raramente en medio (solo en casos específicos)
- NUNCA al inicio de forma aislada

FUNCIÓN:
- NO se leen fonéticamente
- NO contribuyen a la pronunciación
- SOLO indican categoría semántica

CÓMO IDENTIFICAR UN DETERMINATIVO:
----------------------------------

PASO 1: VERIFICAR POSICIÓN
- ¿Está al final de la secuencia? → Muy probable
- ¿Está penúltimo/antepenúltimo? → Posible (puede haber plurales después)
- ¿Está al inicio? → Probablemente NO es determinativo

PASO 2: ANALIZAR CONTEXTO SEMÁNTICO
- ¿El signo tiene sentido semántico con la palabra?
- Ejemplo: Palabra sobre "hablar" + A2 (boca) = TIENE SENTIDO
- Ejemplo: Palabra sobre "guerra" + D54 (piernas) = NO TIENE SENTIDO

PASO 3: VERIFICAR FUNCIÓN FONÉTICA
- ¿El signo es conocido por tener valor fonético?
- Si SÍ y está en medio → Probablemente fonograma
- Si NO tiene función fonética conocida → Probablemente determinativo

PASO 4: CONSULTAR CONOCIMIENTO
- Buscar en la base de determinativos conocidos
- Si se encuentra → DETERMINATIVO CONOCIDO
- Si no se encuentra pero cumple los criterios → DETERMINATIVO DESCONOCIDO
- Si no cumple los criterios → NO ES DETERMINATIVO

EJEMPLOS PRÁCTICOS:
-------------------

Secuencia: "nTr-A40"
→ nTr = fonograma (dios)
→ A40 = al final + figura divina + contexto de "dios"
→ CONCLUSIÓN: A40 es DETERMINATIVO

Secuencia: "ir-ii-D54"
→ ir-ii = fonogramas (venir)
→ D54 = al final + piernas caminando + contexto de movimiento
→ CONCLUSIÓN: D54 es DETERMINATIVO

Secuencia: "mw-N35"
→ mw = fonograma (agua)
→ N35 = al final + ondas de agua + contexto de "agua"
→ CONCLUSIÓN: N35 es DETERMINATIVO

REGLAS CRÍTICAS DEL ESPECIALISTA (V1.5 - Basado en Análisis Real):
-------------------------------------------------------------------------

SIGNOS QUE CASI NUNCA SON DETERMINATIVOS (Evitar falsos positivos):

1. G17 (coruja 'm'): 
   - PRÁCTICAMENTE NUNCA es determinativo
   - Es un fonograma MUY común (marca el sonido /m/)
   - Error identificado: 8 veces en análisis real
   → Si encuentras G17 en posición final, es PROBABLEMENTE un fonograma

2. N35 (agua 'n'):
   - NUNCA es determinativo SOLO
   - Es un fonograma MUY común (marca el sonido /n/)
   - EXCEPCIÓN: N35A (3x N35 secuenciales) SÍ es determinativo de agua
   - Error identificado: 20 veces (¡el más frecuente!)
   → N35 solo = SIEMPRE fonograma
   → N35-N35-N35 (N35A) = DETERMINATIVO de agua

3. S29 (dobla de pano 's'):
   - MUY IMPROBABLE ser determinativo
   - Es un fonograma común (marca el sonido /s/)
   - Error identificado: 10 veces
   → Si encuentras S29, es casi siempre fonograma

4. M17 (junco 'i'):
   - RARAMENTE es determinativo
   - Es un fonograma MUY común (marca sonido /i/ o /y/)
   - Error identificado: 15 veces
   → M17 en posición final = probablemente fonograma

5. D21 (boca 'r'):
   - RARAMENTE es determinativo
   - Es un fonograma común (marca el sonido /r/)
   - Error identificado: 5 veces
   → D21 al final = probablemente fonograma

SIGNOS QUE SON DETERMINATIVOS CONFIABLES (Alta precisión):
- A1 (hombre sentado): 20 identificaciones correctas
- Z1 (trazo vertical - plural): 20 identificaciones correctas
- Z2 (3 trazos - plural): 11 identificaciones correctas
- D54 (piernas caminando): 10 identificaciones correctas
- A40 (dios sentado): 8 identificaciones correctas
- B1 (mujer sentada): 5 identificaciones correctas

REGLA DE ORO V1.5:
Si el signo está en la lista de "CASI NUNCA", se requiere evidencia MUY FUERTE
para considerarlo determinativo. En caso de duda, es probablemente un fonograma.

CASOS ESPECIALES:
-----------------

1. PLURALES DESPUÉS DEL DETERMINATIVO:
   "nTr-A40-Z3" → A40 es determinativo, Z3 es marcador de plural

2. COMPUESTOS:
   Las palabras pueden tener múltiples determinativos
   Ejemplo: "s-Xr-D54-A1" → D54 (movimiento) + A1 (persona)

3. DETERMINATIVOS AUSENTES:
   No todas las palabras tienen determinativos
   Especialmente en textos religiosos o nombres propios

CONSEJO IMPORTANTE:
-------------------
Cuando tenga DUDA, considere:
- Posición en la palabra (peso: 40%)
- Categoría semántica (peso: 30%)
- Función conocida (peso: 20%)
- Patrones similares (peso: 10%)

Si ≥ 60% de los criterios apuntan a determinativo → ES determinativo
"""

# ============================================================================
# DEFINIÇÃO DO GRAFO DE ESTADO DO LANGGRAPH
# ============================================================================

class AgentState(TypedDict):
    """
    Estado compartilhado entre todos os agentes
    
    V1.15 - PARALELIZAÇÃO:
    Campos de INPUT são imutáveis (apenas leitura)
    Campos de OUTPUT usam Annotated para múltiplas escritas seguras
    """
    # ENTRADA (imutáveis - apenas leitura)
    mdc_sequence: str
    image_file: str
    tokens: List[str]
    candidates_info: List[str]
    candidates_text: str
    rule_result: Dict[str, Any]
    expert_context: str
    # V1.35: agent-specific parameter guidance (separate for each agent role)
    semantic_guidance:    str
    structural_guidance:  str
    synthesizer_guidance: str
    enriched_determinatives: Dict[str, Any]
    mdc_structure: Dict[str, Any]
    words: List[Dict[str, Any]]
    
    # RESULTADOS DOS AGENTES (cada agente escreve no seu campo)
    # Não precisa Annotated porque cada agente escreve em campo diferente
    semantic_analysis: str
    structural_analysis: str
    consolidated_analysis: str
    determinatives: List[Dict[str, Any]]
    
    # Raw JSON outputs (end-to-end)
    semantic_json: Dict[str, Any]
    structural_json: Dict[str, Any]
    synthesizer_json: Dict[str, Any]
    # Metadados
    method: str
    success: bool
    error: str

# ============================================================================
# FUNÇÕES DOS AGENTES (NODOS DO GRAFO)
# ============================================================================

def parse_crewai_output(output_text: str, tokens: List[str]) -> List[Dict]:
    """
    VERSÃO CORRIGIDA com debug e parsing robusto
    """
    
    # DEBUG: Mostrar saída do LLM
    print("\n" + "🔍 " + "="*68)
    print("DEBUG: SAÍDA DO LLM (primeiros 600 chars)")
    print("="*70)
    #preview = output_text[:600] if len(output_text) > 600 else output_text
    preview = output_text[:]
    print(preview)
    #if len(output_text) > 600:
    #    print("... (mais texto)")
    print("="*70)
    
    determinatives = []
    
    
    # --------------------------------------------------------------------
    
    # ENHANCED PARSING (v1.15+): handle explicit positions + repetitions
    
    # Tries JSON first (if present), then extracts patterns like:
    
    #   "A1 (pos 5)" / "A1 (position 5)" / "A1 (posição 5)"
    
    # This improves repeated determinatives (same sign multiple times).
    
    # --------------------------------------------------------------------
    
    import json as _json

    
    # (1) Try JSON (best effort)
    
    try:
    
        _txt = output_text.strip()
    
        if "{" in _txt and "}" in _txt:
    
            _start = _txt.find("{")
    
            _end = _txt.rfind("}") + 1
    
            _maybe = _txt[_start:_end]
    
            _data = _json.loads(_maybe)
    
            if isinstance(_data, dict) and isinstance(_data.get("determinatives"), list):
    
                for _d in _data["determinatives"]:
    
                    if not isinstance(_d, dict):
    
                        continue
    
                    _sign = (_d.get("token") or _d.get("sign") or _d.get("code") or "").strip()
    
                    _idx = _d.get("index")
    
                    if not _sign or not isinstance(_idx, int):
    
                        continue
    
                    # accept 0-based; also accept 1-based if matches the tokens
    
                    if 0 <= _idx < len(tokens) and tokens[_idx] == _sign:
    
                        _pos = _idx
    
                    elif 1 <= _idx <= len(tokens) and tokens[_idx - 1] == _sign:
    
                        _pos = _idx - 1
    
                    else:
    
                        continue

    
                    if _sign in DETERMINATIVES_DETAILED:
    
                        info = DETERMINATIVES_DETAILED[_sign]
    
                        determinatives.append({
    
                            'sign': _sign,
    
                            'position': _pos,
    
                            'significado': info.get('significado', info.get('meaning', 'Unknown')),
    
                            'categoria': info.get('categoria', info.get('category', 'unknown')),
    
                            'semantica': info.get('semantica', ''),
    
                            'confidence': 'medium',
    
                            'method': 'langgraph'
    
                        })
    
                    else:
    
                        determinatives.append({
    
                            'sign': _sign,
    
                            'position': _pos,
    
                            'significado': 'not cataloged',
    
                            'categoria': 'unknown',
    
                            'semantica': 'sign not found in local base',
    
                            'confidence': 'low',
    
                            'method': 'langgraph'
    
                        })
    
                if determinatives:
    
                    print(f"📌 Parsed {len(determinatives)} determinatives from JSON output.")
    
    except Exception:
    
        pass

    
    # (2) Position-based extraction (if JSON was not usable)
    
    if not determinatives:
    
        position_patterns = [
    
            r'\b([A-Z]\d+[a-zA-Z]?)\s*\(\s*(?:pos|position|posi[cç][aã]o)\s*[:=]?\s*(\d+)\s*\)',
    
            r'\b([A-Z]\d+[a-zA-Z]?)\s*(?:pos|position|posi[cç][aã]o)\s*[:=]?\s*(\d+)\b',
    
        ]

    
        occurrences = []
    
        for _pat in position_patterns:
    
            for _mm in re.finditer(_pat, output_text, flags=re.IGNORECASE):
    
                occurrences.append((_mm.start(), _mm.group(1), int(_mm.group(2))))
    
        occurrences.sort(key=lambda x: x[0])

    
        if occurrences:
    
            print(f"📌 Found {len(occurrences)} explicit (sign, position) mentions in the LLM output.")
    
            for _, _sign, _pos_raw in occurrences:
    
                _sign = _sign.strip()
    
                if _sign not in tokens:
    
                    continue

    
                # accept 0-based first, then 1-based if token matches
    
                if 0 <= _pos_raw < len(tokens) and tokens[_pos_raw] == _sign:
    
                    _pos = _pos_raw
    
                elif 1 <= _pos_raw <= len(tokens) and tokens[_pos_raw - 1] == _sign:
    
                    _pos = _pos_raw - 1
    
                else:
    
                    # fallback: accept within range (still preserves repetitions)
    
                    if 0 <= _pos_raw < len(tokens):
    
                        _pos = _pos_raw
    
                    elif 1 <= _pos_raw <= len(tokens):
    
                        _pos = _pos_raw - 1
    
                    else:
    
                        continue

    
                if _sign in DETERMINATIVES_DETAILED:
    
                    info = DETERMINATIVES_DETAILED[_sign]
    
                    determinatives.append({
    
                        'sign': _sign,
    
                        'position': _pos,
    
                        'significado': info.get('significado', info.get('meaning', 'Unknown')),
    
                        'categoria': info.get('categoria', info.get('category', 'unknown')),
    
                        'semantica': info.get('semantica', ''),
    
                        'confidence': 'medium',
    
                        'method': 'langgraph'
    
                    })
    
                else:
    
                    determinatives.append({
    
                        'sign': _sign,
    
                        'position': _pos,
    
                        'significado': 'not cataloged',
    
                        'categoria': 'unknown',
    
                        'semantica': 'sign not found in local base',
    
                        'confidence': 'low',
    
                        'method': 'langgraph'
    
                    })
    # Múltiplos padrões para encontrar signos
    patterns = [
        r'[-•]\s*([A-Z]\d+[a-zA-Z]?)\s*[:：]',  # - SIGNO: ou • SIGNO:
        r'([A-Z]\d+[a-zA-Z]?)\s*\(',  # SIGNO (
        r'([A-Z]\d+[a-zA-Z]?)\s*-',   # SIGNO -
    ]
    
    found_signs = set()
    for pattern in patterns:
        matches = re.findall(pattern, output_text)
        found_signs.update(matches)
    
    # Se não encontrou nada com padrões específicos, buscar todas menções
    if not found_signs:
        print("⚠️  Nenhum padrão específico, buscando todas as menções...")
        found_signs = set(re.findall(r'\b([A-Z]\d+[a-zA-Z]?)\b', output_text))
    
    print(f"\n📌 Signos encontrados no texto do LLM: {sorted(found_signs)}")
    print(f"📌 Tokens da sequência: {tokens}")
    
    # Validar cada signo
    for sign in sorted(found_signs):
        sign = sign.strip()
        
        # Verificar se está nos tokens
        if sign not in tokens:
            print(f"   ⏭️  {sign}: NÃO está na sequência, pulando")
            continue
        
        # Verificar se está na base de conhecimento
        if sign not in DETERMINATIVES_DETAILED:
            print(f"   ⚠️  {sign}: NÃO está na base de conhecimento")
            # Adicionar mesmo assim como "desconhecido"
            determinatives.append({
                'sign': sign,
                'position': tokens.index(sign),
                'significado': 'não catalogado',
                'categoria': 'unknown',
                'semantica': 'signo não identificado na base',
                'confidence': 'low',
                'method': 'langgraph'
            })
            print(f"   ➕ {sign}: adicionado como 'desconhecido' (pos {tokens.index(sign)})")
            continue
        
        # Adicionar da base de conhecimento
        info = DETERMINATIVES_DETAILED[sign]
        det = {
            'sign': sign,
            'position': tokens.index(sign),
            'significado': info['significado'],
            'categoria': info['categoria'],
            'semantica': info['semantica'],
            'confidence': 'high',
            'method': 'langgraph'
        }
        determinatives.append(det)
        print(f"   ✅ {sign}: {info['significado']} (pos {det['position']})")
    
    print(f"\n📊 RESULTADO INICIAL: {len(determinatives)} determinativos identificados")
    
    # ========================================================================
    # V1.6: VALIDAÇÃO COMPLETA (Precisão + Recall)
    # ========================================================================
    print("\n🔍 APLICANDO VALIDAÇÃO V1.6 (Completa: Precisão + Recall)...")
    print("="*70)
    
    # Construir sequência para contexto (junta tokens)
    sequence_context = '-'.join(tokens)
    
    # Validar com regras completas v1.6 (excluded_signs + priority_signs + padrões)
    validated_dets = validate_determinatives_v16(determinatives, tokens, sequence_context)
    
    print(f"\n📊 APÓS VALIDAÇÃO V1.6: {len(validated_dets)} determinativos validados")
    print(f"   Mudança: {len(validated_dets) - len(determinatives):+d} determinativos")
    print("="*70 + "\n")
    
    return validated_dets



# ============================================================================
# 📝 NOTA: PROMPTS MELHORADOS DISPONÍVEIS
# ============================================================================
# Os prompts dos agentes foram melhorados para:
# 1. Aceitar repetições de determinativos (Aa2-Aa2-Aa2)
# 2. Analisar TODA a sequência (não só o final)
# 3. Identificar múltiplas palavras
# 4. Usar categorias semânticas expandidas
# 
# Para aplicar os prompts melhorados completos, consulte:
# PROMPTS_MELHORADOS.py
# ============================================================================


def word_boundary_agent_node(state: AgentState) -> AgentState:
    """
    Deterministic pre-step that infers probable word/compound boundaries and
    generates CONTIGUOUS SUFFIX span candidates per word (determinative blocks).

    This aligns with Egyptological practice:
      - determinatives tend to be final of the word/compound, not necessarily final of the whole sequence.
    """
    try:
        toks = state.get("tokens", []) or []
        mdc_struct = state.get("mdc_structure", {}) or {}
        words = build_word_boundaries(toks, mdc_struct)
        state["words"] = words
        state["word_span_candidates"] = generate_suffix_span_candidates(words, toks, max_len=6)
        # Phonetic-complement hints (Middle Egyptian): common uniliterals near word end are often complements, not determinatives.
        state["phonetic_complement_hints"] = build_phonetic_complement_hints(toks, words)
        state["phonetic_complement_pairs"] = {(h["token"], int(h["index"])) for h in state.get("phonetic_complement_hints", []) if isinstance(h, dict)}
    except Exception:
        state["words"] = []
        state["word_span_candidates"] = []
    return state



def semantic_agent_node(state: AgentState) -> AgentState:
    """
    Nodo do agente semanticista (análise semântica e cultural)
    Usa os MESMOS PROMPTS do CrewAI
    """
    if not USE_LANGGRAPH:
        state['semantic_analysis'] = ""
        return state
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Configurar LLM (mesma lógica do CrewAI)
        llm = None
        if 'OPENAI_API_KEY' in os.environ:
            llm = ChatOpenAI(
                model="gpt-5.4-nano",
                temperature=0.0,
                top_p=1,
                seed=GLOBAL_SEED,
                api_key=os.environ.get('OPENAI_API_KEY')
            )
        elif 'ANTHROPIC_API_KEY' in os.environ:
            # v1.42: temperature=0.0 for reproducibility.
            # Sign identification is a classification task — no creative variance needed.
            # Note: Anthropic has no seed= param; temp=0.0 is the best available fix.
            llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.0,
                api_key=os.environ.get('ANTHROPIC_API_KEY')
            )
        
        if llm is None:
            state['semantic_analysis'] = ""
            return state
        
        # Usar enriched_determinatives do state (criado antes dos agentes)
        enriched_determinatives = state.get('enriched_determinatives', DETERMINATIVES_DETAILED)
        
        determinatives_info = "\n".join([
            f"• {sign}: {info.get('significado', info.get('meaning', 'Unknown'))} (cat: {info.get('categoria', info.get('category', 'Unknown'))})"
            for sign, info in sorted(enriched_determinatives.items())
        ])
                
        system_prompt = (
        """You are an Egyptologist specialized in SEMANTIC analysis of Egyptian MdC (Manuel de Codage) sequences.

        GOAL: Identify determinatives with high precision using semantic/cultural evidence.
        
        
        MANDATORY RULES:
        1) Use ONLY the provided tokens; NEVER invent signs.
        2) Never output separators as determinatives: '-', ':', '*'.
        3) Prefer precision over quantity. If evidence is weak, output an empty list.
        4) If a determinative occurs multiple times, list EACH occurrence separately with its token index (0-based).
        5) Output MUST be valid JSON only (no markdown, no extra text).
        6) V1.37 QUALITY OVER QUANTITY: Prefer accuracy over completeness.
           Output ONLY signs you are confident about. A correct partial prediction
           is better than a complete prediction with errors.
           NEVER add uncertain signs to reach any minimum count.
           It is perfectly correct to output only 1 determinative even in a 50-sign sequence.

        V1.33 CRITICAL DISAMBIGUATION RULES:

        ⚠️ V1.37 CRITICAL: EVALUATE EACH SIGN INDEPENDENTLY
        The presence of phonogram signs (I9, L1, I10, D58) in a sequence does NOT
        make other signs in that sequence less likely to be determinatives.
        Rejecting I9 or L1 is a LOCAL decision for that specific sign ONLY.
        After rejecting I9, continue evaluating A1, Z1, A53, Y2, etc. normally.
        Do NOT let high-risk signs cause you to return empty for the whole sequence.

        O1 (HOUSE PLAN — building) — V1.38 POSITION-DEPENDENT FIX:
           ❌ REJECT when O1 is at index 0 or 1 (first signs of sequence) — it IS the
              grammatical subject "house/estate/domain". Not a classifier here.
           ✅ ACCEPT as possible determinative when O1 appears at index ≥ 2 (mid or end
              of sequence, after phonetic core). In this position it classifies estate/domain.
           NOTE: "M23-G43-O1-D21-D54..." → O1 at index 2 IS a determinative here.
                 "O1-Z1-M17-A1-..." → O1 at index 0 is the subject, REJECT.

        A1 (SEATED MAN — person) — V1.38 LOGOGRAM VS DETERMINATIVE:
           ✅ DETERMINATIVE when A1 appears AFTER the phonetic core of a word in
              final/near-final position within its word group.
           ❌ LOGOGRAM (REJECT) when:
              • A1 is the ONLY sign in a short sequence (2-5 tokens) — it IS the word
              • A1 appears at the START of a word group (before phonetic signs)
              • The sequence begins with D35 (negation) — see below
           NOTE: When you are UNSURE whether A1 is logogram or determinative, prefer
                 its known determinative neighbours (D54, Z1, A40) and evaluate A1
                 based on whether those neighbours are also determinatives.

        D35 (NEGATION — arm in gesture of refusal) — V1.38 NEW:
           ⚠️ When D35 appears, especially near the START of the sequence, this is a
           NEGATIVE CONSTRUCTION. In negated phrases, human and object signs
           (A1, A24, D4, etc.) typically serve phonetic/logographic roles, NOT as
           determinatives. Reduce confidence in ALL determinative predictions.
           If GT would be empty for a negative sequence, that is CORRECT — output [].

        D54 (WALKING LEGS — action):
           ✅ DETERMINATIVE only when the word explicitly means: go, walk, come, travel, enter, run, procession.
           ❌ REJECT in: work actions (build, write, fight), human descriptions, general activities.
           ❌ REJECT when A1/A24 already classifies the person and no motion word is present.
           NOTE: D54 + A1 can coexist when the text means "a man who goes/walks".

        T18 (STAFF/CROOK — object):
           ✅ DETERMINATIVE only for: follower, adherent, companion ("Xr" = under/following).
           ❌ REJECT in general human activity contexts — often a logogram, NOT a determinative.

        V1 (LASSO — object):
           DEFAULT = phonogram or numeral (100,000). NOT a determinative.
           ✅ Only accept in explicit binding/tying context with strong evidence.
           ❌ REJECT unless clear binding semantic is present.

        X3 / X6 (BREAD SIGNS — food) — V1.38 NEW:
           ✅ Only determinatives in EXPLICIT OFFERING LISTS with multiple food items.
           ❌ REJECT in general ritual sequences where bread is mentioned but not the main classifier.
           Test: are there other food/offering signs (X4, W24, W22) nearby? If not, be cautious.

        E21 (DOG/JACKAL — animal) — V1.38 NEW:
           ✅ Only determinative in explicit CANINE or ANUBIS-WORSHIP context.
           ❌ REJECT as generic animal classifier for non-canine animals.
           Test: does the sequence contain Anubis-related words or explicit dog/jackal reference?

        N16 vs N24 (LAND SIGNS — easily confused):
           N16 = DRY cultivated field/land (agricultura, campo seco).
           N24 = POOL/canal with water (piscina, estanque, agua).
           Do NOT predict N16 when the context is about water, pools or canals.
           Do NOT predict N24 when the context is about dry fields or agriculture.

        G39 (SWALLOW/DUCK BIRD — animal):
           Easily confused with H8 (duck egg) and A47 (shepherd with staff).
           G39 is often a LOGOGRAM for the word "swallow/duck", NOT a determinative.
           Only accept as determinative in explicit bird/waterfowl context.
           When uncertain between G39 and a similar sign, prefer the one in the KB candidates list.
        
        
        
        EGYPTOLOGICAL PRINCIPLES (Middle Egyptian):
        A) Mixed system: each sign may function as (i) phonogram (sound), (ii) logogram/ideogram (word/idea), or (iii) determinative/classifier (NOT read; semantic disambiguation). You must output determinatives/classifiers only.
        
        B) Phonetic complements (CRITICAL): common uniliteral phonograms near the end of a word (e.g., M17, D21, S29, G17, N35, X1) are very often phonetic complements, not determinatives. Treat them as phonograms/complements by DEFAULT unless there is strong classifier evidence.
        
        C) CRITICAL CONTEXT-DEPENDENT SIGNS (v1.27 - Common False Positive Patterns):
        
        C1) VESSEL/CONTAINER SIGNS (W22, W24, W23):
           ⚠️ These ARE VALID determinatives - but ONLY in vessel/liquid/offering contexts!
           
           VALID CONTEXTS:
           • Near food/offering signs: bread (X4), beer, wine, oil, offerings
           • Semantic field: vessel, jug, pot, container, liquid, water, offering
           • Multiple W24 in sequence (e.g., W24-W24-Z3)
           • Near other container determinatives (Aa1, W9)
           
           REJECT IF:
           • No vessel/liquid/offering semantic content
           • Isolated occurrence without supporting context
           • Sequence about human actions (walking, speaking) not offerings
           
           EXAMPLE VALID: "offerings of bread X4, beer W24, wine" → W24 is determinative ✓
           EXAMPLE INVALID: "man walking to the house" + W24 → W24 rejected ✗
        
        C2) DIVINE/SACRED SIGNS (R8, R7):
           ⚠️ These ARE VALID determinatives - but ONLY in divine/sacred contexts!
           
           VALID CONTEXTS:
           • Near deity signs: A40 (god), B1 (goddess), C12 (deity seated)
           • Semantic field: god, goddess, divine, sacred, temple, worship, praise, deity
           • Religious/ritual context
           
           REJECT IF:
           • No divine/sacred semantic content
           • Secular/mundane actions
           • Human subjects without divine association
           
           EXAMPLE VALID: "praising god A40 Seth" + R8 → R8 is determinative ✓
           EXAMPLE INVALID: "walking to the palace" + R8 → R8 rejected ✗
        
        C3) BODY PART SIGNS (D4=eye, F4=bull head, F51=flesh):
           • D4 (eye): ONLY for vision/seeing concepts (ver, contemplar, vigilar)
           • F4 (bull head): ONLY for bovine/cattle OR "parte delantera" concepts
           • F51 (flesh/blood): Always valid for body/butchering contexts
           • Higher confidence threshold required (>0.85)
        
        C3a) ⚠️ V1.28 CRITICAL: ANIMAL/REPTILE SIGNS (I9, I10 - HIGHLY AMBIGUOUS!):
           🔴 I9, I10 (cobras/vipers) caused +65 FP in v1.27 - EXTREME CAUTION REQUIRED!
           
           These signs are EXTREMELY ambiguous (fonograma vs determinativo).
           DEFAULT treatment: PHONOGRAM unless OVERWHELMING evidence.
           
           ONLY determinative IF:
           • Explicit animal/reptile/snake semantic context
           • Near other animal determinatives (E1, E23, I14)
           • Words about: snake, serpent, cobra, venom, reptile, dangerous animal
           • Very high confidence (>0.90)
           
           REJECT IF:
           • Isolated occurrence
           • No animal/reptile context
           • Generic position without semantic support
           • ANY doubt → treat as phonogram
           
           EXAMPLE VALID: "killed the serpent I9" → I9 is determinative (snake context) ✓
           EXAMPLE INVALID: isolated "I9" without animal context → I9 is phonogram ✗
        
        C3b) ⚠️ V1.28 CRITICAL: INSECT/SACRED SIGNS (L1 - SCARAB BEETLE):
           🔴 L1 (escaravelho) caused +22 FP in v1.27 - TREAT LIKE R8 (divine context)!
           
           L1 is a SACRED symbol but also used phonetically.
           Requires divine/religious/sacred context like R8.
           
           ONLY determinative IF:
           • Divine/sacred/religious context
           • Near deity signs (A40, B1, R8)
           • Words about: sacred, amulet, divine, beetle, scarab, protection
           • High confidence (>0.85)
           
           REJECT IF:
           • No religious/sacred context
           • Secular contexts
           • Isolated without divine associations
           
           EXAMPLE VALID: "sacred scarab amulet L1" → L1 is determinative ✓
           EXAMPLE INVALID: isolated "L1" without sacred context → L1 is phonogram ✗
        
        C3c) TOOL/AGRICULTURE SIGNS (U1=sickle, U6=hoe):
           • ONLY in agriculture/tool/harvest context
           • Words about: sickle, harvest, farming, plow, agriculture, tool
           • Higher confidence threshold (>0.78)
        
        C3d) BODY ACTION SIGNS (D58=leg advancing, D56=leg):
           • ONLY in movement/action context
           • Words about: walk, go, move, advance, travel, leg, foot
           • Higher confidence threshold (>0.80)
        
        C4) PHONOGRAM-PRIMARY SIGNS (mostly NOT determinatives):
           • M17, D21, S29, G17, N35, X1, G43: DEFAULT = phonetic complements
           • Only promote to determinative with OVERWHELMING evidence
           • X1 as determinative: only in food/offering context + high confidence
        
        D) Unknown-sign placeholders (CRITICAL): tokens that look like OCR/unknown placeholders (e.g., starting with "US") MAY be logograms or determinative/classifier placeholders. If such a token repeats (≥2) AND appears in word-final / tail positions within a block/word, treat it as a plausible determinative/classifier with needs_lookup=true (do NOT automatically demote it to phonetic complement). If still uncertain, include it as an ALTERNATIVE with a clear promotion condition.
        
        E) Word-final layering heuristic: reason in layers — (1) phonetic/logographic core, (2) phonetic complements near the end, (3) determinatives/classifiers, (4) plural strokes (Z1/Z2/Z3). Only layer (3) are semantic classifiers; plural strokes are functional markers.
        
        F) Alternatives must be actionable: for each alternative, include the minimal PROMOTION CONDITION under which it would replace MAIN (write it inside the 'reason' field).
        
        G) When uncertain: prefer ALTERNATIVE over weak MAIN. Better to flag ambiguity than make weak assertion.
        
        
        
        REFERENCE (determinatives in this sequence):
        """
            + determinatives_info
            + """
        
        
        
        OUTPUT FORMAT (STRICT):
        BEGIN_JSON
        {...the JSON object exactly matching the schema...}
        END_JSON
        
        Return ONLY what is between BEGIN_JSON and END_JSON (no extra text).
        
        OUTPUT JSON SCHEMA:
        {
          "main_interpretation": {
            "semantic_classifiers": [
              {"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}
            ],
            "functional_markers": [
              {"token":"Z1","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}
            ],
            "determinatives": [
              {"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}
            ],
            "per_word": [
              {
                "word_id": 0,
                "semantic_classifiers": [{"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true}],
                "functional_markers": [{"token":"Z1","index":0,"gardiner_code":null,"needs_lookup":true}]
              }
            ],
            "justification": "Spanish text (may be long)",
            "validation": "Spanish text (optional)"
          },
          "alternative_interpretations": [
            {
              "semantic_classifiers": [{"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}],
              "functional_markers": [{"token":"Z1","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}],
              "determinatives": [{"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}],
              "perspective": "Semántica|Estructural|Reglas|Mixta",
              "reason": "Spanish text"
            }
          ],
          "ambiguity": {
            "factors": ["Spanish bullet", "..."],
            "consensus": "Spanish text"
          },
          "recommendation": ["Spanish bullet", "..."]
        }
        """
        )


        human_prompt = f"""EXPERT CONTEXT (examples & criteria):
{state['expert_context']}

SEMANTIC CALIBRATION GUIDANCE (data-validated, apply to meaning-based decisions):
{state.get('semantic_guidance', '')}

SEQUENCE (MdC):
{state['mdc_sequence']}

TOKENS WITH INDICES (0-based, includes separators):
{list(enumerate(state['tokens']))}

MdC STRUCTURE HINTS (computed) + WORD UNITS:
PHONETIC COMPLEMENT HINTS (computed; treat as complements by default):
{state.get('phonetic_complement_hints', [])}

- block_final_token_indices: {state.get('mdc_structure', {}).get('block_final_token_indices', [])}
- blocks: {state.get('mdc_structure', {}).get('blocks', [])}
- groups: {state.get('mdc_structure', {}).get('groups', [])}

LOCAL CANDIDATES (meaning hints):
{state['candidates_text']}

RULE-BASED BASELINE (validated):
{state['rule_result']}

TASK:
- Identify determinatives supported by semantic/cultural evidence.
- Use 'local_candidate' when the token appears in the local knowledge base.
- Use 'rule_support' when baseline strongly supports the same token+index.
- Use 'hypothesis' only if plausible but not strongly supported.
- Return JSON ONLY, strictly matching the schema.
"""

        # Invocar LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = llm.invoke(messages)
        _raw = response.content if hasattr(response, 'content') else str(response)
        state['semantic_raw'] = _raw
        semantic_data = parse_llm_json(_raw)
        state['semantic_json'] = semantic_data
        state['semantic_analysis'] = semantic_data.get('semantic_analysis', '') if semantic_data else ''
        
        semantic_candidates = get_candidates_from_json(semantic_data) if semantic_data else []
        print(f"✅ Semantic JSON parsed: {len(semantic_candidates)} candidates")
        if len(semantic_candidates) == 0:
            raw = state.get('semantic_raw','')
            if isinstance(raw, str) and raw.strip():
                print('   ℹ️ semantic raw head:', raw.strip().replace('\n',' ')[:300])
            else:
                print('   ℹ️ semantic raw is empty')
        
        print("✅ Análise semântica concluída")
        
    except Exception as e:
        print(f"⚠️  Erro no agente semanticista: {e}")
        state['semantic_analysis'] = f"Error: {str(e)}"
    
    return state


def structural_agent_node(state: AgentState) -> AgentState:
    """
    Nodo do agente estruturalista (análise estrutural e posicional)
    
    V1.15 - VERSÃO INDEPENDENTE (PARALELIZAÇÃO):
    • NÃO usa state['semantic_analysis']
    • Executa SIMULTANEAMENTE com agente semântico
    • Usa apenas: mdc_sequence, candidates_text, rule_result, expert_context
    • Foco em padrões posicionais e estruturais
    """
    if not USE_LANGGRAPH:
        state['structural_analysis'] = ""
        return state
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Configurar LLM
        llm = None
        if 'OPENAI_API_KEY' in os.environ:
            llm = ChatOpenAI(
                model="gpt-5.4-nano",
                temperature=0.0,
                top_p=1,
                seed=GLOBAL_SEED,
                api_key=os.environ.get('OPENAI_API_KEY')
            )
        elif 'ANTHROPIC_API_KEY' in os.environ:
            llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.0,
                api_key=os.environ.get('ANTHROPIC_API_KEY')
            )
        
        if llm is None:
            state['structural_analysis'] = ""
            return state
        
        # Criar descrição dos determinativos
        # Enriquecer lista com busca Gardiner
        # enriched_determinatives = enrich_determinatives_with_lookup(state['tokens'], DETERMINATIVES_DETAILED)
        
        # determinatives_info = "\n".join([
        #     f"• {sign}: {info.get('significado', info.get('meaning', 'Unknown'))} (cat: {info.get('categoria', info.get('category', 'Unknown'))})"
        #     for sign, info in sorted(enriched_determinatives.items())
        # ])
        # PROMPT (STRUCTURAL) — JSON end-to-end (v3.0)
        system_prompt = """You are an Egyptologist specialized in STRUCTURAL analysis of Egyptian MdC sequences.

GOAL: Identify determinatives using positional/structural evidence (without semantic interpretation).

MANDATORY RULES:
1) Use ONLY the provided tokens; NEVER invent signs.
2) Never output separators as determinatives: '-', ':', '*'.
3) Provide explicit structural evidence (end-of-unit positions, stacks ':', ligatures '*', repeated patterns, etc.).
4) Prefer precision: if evidence is weak, output an empty list.
5) Output MUST be valid JSON only (no markdown, no extra text).

V1.33 DISAMBIGUATION (apply to structural analysis too):
- O1 (house plan) V1.38 FIX — POSITION-DEPENDENT:
  O1 at index 0-1 = logogram (grammatical subject), REJECT as determinative.
  O1 at index >= 2 (after phonetic core) = possible building/estate determinative, evaluate normally.
- A1 (seated man) V1.38: Only determinative when structurally FINAL within its word group.
  A1 at the beginning of a word group = logogram, not determinative.
  A1 in a short 2-5 sign sequence = usually the subject, not a classifier.
- D35 (negation arm): If D35 appears near index 0-2, this is a negative construction.
  Structurally, negative phrases rarely contain determinatives. Reduce all predictions.
- D54 (walking legs): only accept as determinative when structurally final in a motion-verb word group.
  If the word group contains no motion semantics, reject D54 even at word-final position.
- T18 (staff): structurally final position alone is insufficient — requires follower/companion context.
- V1 (lasso): default = phonogram/numeral regardless of position. Strong binding context required.
- N16 vs N24: N16 = dry field, N24 = water pool. Position alone cannot distinguish them.
  If the word is about water/pool → N24. If about dry land/agriculture → N16.
- G39 (swallow): often a logogram at word-initial position, not a determinative.
  Only accept structurally when it occupies a clear word-final classifier position.



EGYPTOLOGICAL PRINCIPLES (Middle Egyptian):
A) Mixed system: each sign may function as (i) phonogram (sound), (ii) logogram/ideogram (word/idea), or (iii) determinative/classifier (NOT read; semantic disambiguation). You must output determinatives/classifiers only.
B) Phonetic complements (CRITICAL): common uniliteral phonograms near the end of a word (e.g., M17, D21, S29, G17, N35, X1) are very often phonetic complements, not determinatives. Treat them as phonograms/complements by DEFAULT unless there is strong classifier evidence.
C) Word-final layering heuristic: reason in layers — (1) phonetic/logographic core, (2) phonetic complements near the end, (3) determinatives/classifiers, (4) plural strokes (Z1/Z2/Z3). Only layer (3) are semantic classifiers; plural strokes are functional markers.
D) Alternatives must be actionable: for each alternative, include the minimal PROMOTION CONDITION under which it would replace MAIN (write it inside the 'reason' field).


OUTPUT FORMAT (STRICT):
BEGIN_JSON
{...the JSON object exactly matching the schema...}
END_JSON

Return ONLY what is between BEGIN_JSON and END_JSON (no extra text).

OUTPUT JSON SCHEMA:
{
  "main_interpretation": {
    "semantic_classifiers": [
      {"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}
    ],
    "functional_markers": [
      {"token":"Z1","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}
    ],
    "determinatives": [
      {"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}
    ],
    "per_word": [
      {
        "word_id": 0,
        "semantic_classifiers": [{"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true}],
        "functional_markers": [{"token":"Z1","index":0,"gardiner_code":null,"needs_lookup":true}]
      }
    ],
    "justification": "Spanish text (may be long)",
    "validation": "Spanish text (optional)"
  },
  "alternative_interpretations": [
    {
      "semantic_classifiers": [{"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}],
      "functional_markers": [{"token":"Z1","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}],
      "determinatives": [{"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}],
      "perspective": "Semántica|Estructural|Reglas|Mixta",
      "reason": "Spanish text"
    }
  ],
  "ambiguity": {
    "factors": ["Spanish bullet", "..."],
    "consensus": "Spanish text"
  },
  "recommendation": ["Spanish bullet", "..."]
}
"""

        # V1.35: replaced dead duplicate human_prompt; now uses structural_guidance
        human_prompt = f"""EXPERT CONTEXT (examples & criteria):
{state['expert_context']}

STRUCTURAL CALIBRATION GUIDANCE (data-validated, apply to position-based decisions):
{state.get('structural_guidance', '')}

STRUCTURAL ANALYSIS — evaluate multiple criteria

SEQUENCE: {state['mdc_sequence']}
TOKENS WITH POSITIONS (0-based): {list(enumerate(state['tokens']))}

MdC STRUCTURE HINTS:
- block_final_token_indices: {state.get('mdc_structure', {}).get('block_final_token_indices', [])}
- blocks: {state.get('mdc_structure', {}).get('blocks', [])}
- groups: {state.get('mdc_structure', {}).get('groups', [])}

CANDIDATES:
{state['candidates_text']}

RULE-BASED BASELINE (validated):
{state['rule_result']}

TASK:
1) Criterion A (strict positional): end-of-word / end-of-block signals.
2) Criterion B (pattern-based): plurals (Z1/Z2/Z3), DET-PLURAL ([SIGN]-[PLURAL]), repeated patterns.
3) Criterion C (compound words): allow multiple determinatives across multiple words.

CRITICAL RULE — MULTIPLE WORDS & REPETITIONS:
A sequence can contain multiple words, each with its own determinative.
The same determinative may occur multiple times.
List EACH occurrence with its position; never collapse into "SIGN (3x)".

If you believe there are no determinatives, write exactly:
DETERMINATIVES: []
"""


        # Invocar LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = llm.invoke(messages)
        _raw = response.content if hasattr(response, 'content') else str(response)
        state['structural_raw'] = _raw
        structural_data = parse_llm_json(_raw)
        state['structural_json'] = structural_data
        state['structural_analysis'] = structural_data.get('structural_analysis', '') if structural_data else ''
        
        structural_candidates = get_candidates_from_json(structural_data) if structural_data else []
        print(f"✅ Structural JSON parsed: {len(structural_candidates)} candidates")
        if len(structural_candidates) == 0:
            raw = state.get('structural_raw','')
            if isinstance(raw, str) and raw.strip():
                print('   ℹ️ structural raw head:', raw.strip().replace('\n',' ')[:300])
            else:
                print('   ℹ️ structural raw is empty')
        
        print("✅ Análise estrutural concluída")
        
    except Exception as e:
        print(f"⚠️  Erro no agente estruturalista: {e}")
        state['structural_analysis'] = f"Error: {str(e)}"
    
    return state


def synthesizer_agent_node(state: AgentState) -> AgentState:
    """
    Nodo do agente sintetizador (síntese e decisão final)
    
    V1.15 - AGUARDA AMBAS ANÁLISES (PARALELIZAÇÃO):
    • LangGraph automaticamente espera semantic_analysis E structural_analysis
    • Sintetizador só executa quando AMBOS agentes terminam
    • Integra as duas perspectivas independentes
    """
    if not USE_LANGGRAPH:
        state['consolidated_analysis'] = ""
        state['determinatives'] = []
        return state
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Configurar LLM
        llm = None
        if 'OPENAI_API_KEY' in os.environ:
            llm = ChatOpenAI(
                model="gpt-5.4-nano",
                temperature=0.0,
                top_p=1,
                seed=GLOBAL_SEED,
                api_key=os.environ.get('OPENAI_API_KEY')
            )
        elif 'ANTHROPIC_API_KEY' in os.environ:
            llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.0,
                api_key=os.environ.get('ANTHROPIC_API_KEY')
            )
        
        if llm is None:
            state['consolidated_analysis'] = ""
            state['determinatives'] = []
            return state
        # PROMPT (SYNTHESIZER) — three-agent synthesis v1.39
        system_prompt = """You are a senior Egyptologist acting as the FINAL ARBITER.
You receive outputs from THREE INDEPENDENT AGENTS. Treat each as a peer expert.

═══════════════════════════════════════════════════════════════
AGENT PROFILES AND RELIABILITY
═══════════════════════════════════════════════════════════════

AGENT 1 — SEMANTIC (Semanticista):
  Strength: understands Egyptian cultural meaning and word semantics.
  Best at: context-dependent signs (W24 in food context, R8 in divine context,
           D54 in motion context, D4 for vision words).
  Weakness: can miss signs it has no KB entry for; sometimes over-rejects.

AGENT 2 — STRUCTURAL (Estruturalista):
  Strength: understands MdC syntax, positions, word groups, plural markers.
  Best at: word-final signs, Z1/Z2/Z3 plural markers, positional patterns.
  Weakness: cannot judge semantic context; may accept signs in wrong context.

AGENT 3 — RULE-BASED (87.0% validated macro-F1, deterministic):
  Strength: parameter-optimised on 80 training sequences; highest aggregate F1.
  Best at: reliable signs (A1, Z1, A40, B1, A24) in final positions;
           signs with validated context thresholds (W24, R8, D4).
  Weakness: only knows signs in its knowledge base; cannot handle novel signs.

═══════════════════════════════════════════════════════════════
THREE-AGENT VOTING RULES (apply in this order)
═══════════════════════════════════════════════════════════════

RULE 1 — ALL THREE AGREE → ACCEPT immediately (highest confidence).
  Any sign predicted by Semantic + Structural + Rules: accept without hesitation.

RULE 2 — AGENT 3 (RULES) + ANY ONE LLM AGENT → STRONG EVIDENCE, accept.
  Rules confirmed sign AND at least one of {Semantic, Structural} also supports it:
  → Accept the sign. Rules have 87% validated F1; their agreement is strong signal.
  Exception: high-risk signs (I9, L1, I10) require BOTH LLM agents + rules.

RULE 3 — BOTH LLM AGENTS AGREE, RULES ABSTAIN/REJECT → MODERATE EVIDENCE.
  Semantic + Structural agree but Rules did not predict it:
  → Accept if: the sign is a RELIABLE sign (A1, Z1, A40, B1, A24, N1) in final position.
  → Accept if: the sign is a known context-sign (W24, R8, D4) AND Semantic confirmed context.
  → Evaluate carefully for other signs: prefer to accept than to miss a determinative.
  → REJECT only if there is an explicit reason (phonogram, logogram, negation context).

RULE 4 — ONLY ONE AGENT SUPPORTS → WEAK EVIDENCE.
  → Accept if: reliable sign (A1, Z1, A40) + final position + the supporting agent is Rules.
  → Accept if: the sign appears in KB candidates AND is in word-final position.
  → Otherwise: place in ALTERNATIVES, do NOT include in MAIN.

RULE 5 — ALL THREE DISAGREE OR ALL ABSTAIN → OUTPUT EMPTY [].
  If no sign has multi-source support AND the sequence has no strong candidates:
  → Output empty determinative list.
  → This is correct — not all sequences have determinatives.

═══════════════════════════════════════════════════════════════
MANDATORY RULES
═══════════════════════════════════════════════════════════════
1) Use ONLY tokens present in the provided token list; NEVER invent signs.
2) Never output separators as determinatives: '-', ':', '*'.
3) Repetitions matter: include EACH occurrence separately with its token index (0-based).
4) Prioritize precision; do NOT force determinatives when evidence is weak.
5) Output MUST be valid JSON only (no markdown, no extra text).
6) Determinatives are typically final in a word/compound but may appear non-finally
   in formulaic or enumerative sequences when all three agents confirm.

OUTPUT LANGUAGE: All narrative fields in SPANISH. Sign codes unchanged.

OUTPUT JSON SCHEMA:
{
  "main_interpretation": {
    "semantic_classifiers": [
      {"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}
    ],
    "functional_markers": [
      {"token":"Z1","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}
    ],
    "determinatives": [
      {"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}
    ],
    "per_word": [
      {
        "word_id": 0,
        "semantic_classifiers": [{"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true}],
        "functional_markers": [{"token":"Z1","index":0,"gardiner_code":null,"needs_lookup":true}]
      }
    ],
    "justification": "Spanish text — state which rule applied and which agents agreed",
    "validation": "Spanish text (optional)"
  },
  "alternative_interpretations": [
    {
      "semantic_classifiers": [{"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}],
      "functional_markers": [{"token":"Z1","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}],
      "determinatives": [{"token":"SIGN","index":0,"gardiner_code":null,"needs_lookup":true, "word_id": 0}],
      "perspective": "Semántica|Estructural|Reglas|Mixta",
      "reason": "Spanish text — include which voting rule would promote this to MAIN"
    }
  ],
  "ambiguity": {
    "factors": ["Spanish bullet", "..."],
    "consensus": "Spanish text — state the agent agreement level (all-3, 2-of-3, 1-of-3)"
  },
  "recommendation": ["Spanish bullet", "..."]
}
"""

        # Format rule-based output as Agent 3 structured input
        _rule_dets  = state['rule_result'].get('determinatives', []) \
                      if isinstance(state.get('rule_result'), dict) else []
        def _safe_conf(v):
            if isinstance(v, (int, float)): return round(float(v), 3)
            if isinstance(v, str):
                return {'low': 0.3, 'medium': 0.6, 'high': 0.9,
                        'very_high': 1.0, 'very high': 1.0}.get(v.lower(), 0.5)
            return 0.0
        _rule_signs = [{'sign': d.get('sign',''), 'index': d.get('index',-1),
                        'confidence': _safe_conf(d.get('confidence', 0.0))}
                       for d in _rule_dets if isinstance(d, dict) and d.get('sign')]
        _rule_empty = len(_rule_signs) == 0

        human_prompt = f"""THREE-AGENT SYNTHESIS — apply voting rules from system prompt

SYNTHESIS DECISION GUIDANCE (data-validated — apply for final arbitration):
{state.get('synthesizer_guidance', '')}

SEQUENCE (MdC):
{state['mdc_sequence']}

TOKENS WITH INDICES (0-based, includes separators):
{list(enumerate(state['tokens']))}

MdC STRUCTURE HINTS:
- block_final_token_indices: {state.get('mdc_structure', {}).get('block_final_token_indices', [])}
- blocks: {state.get('mdc_structure', {}).get('blocks', [])}
- groups: {state.get('mdc_structure', {}).get('groups', [])}
- words (probable word/compound units): {state.get('words', [])}
- suffix span candidates per word (determinative blocks): {state.get('word_span_candidates', [])}

══════════════════════════════════════════════════════════
AGENT 1 — SEMANTIC OUTPUT (Semanticista):
══════════════════════════════════════════════════════════
{json.dumps(state.get('semantic_json', {}), ensure_ascii=False)}

══════════════════════════════════════════════════════════
AGENT 2 — STRUCTURAL OUTPUT (Estruturalista):
══════════════════════════════════════════════════════════
{json.dumps(state.get('structural_json', {}), ensure_ascii=False)}

══════════════════════════════════════════════════════════
AGENT 3 — RULE-BASED OUTPUT (87.0% validated macro-F1, deterministic):
══════════════════════════════════════════════════════════
Predicted determinatives: {_rule_signs if not _rule_empty else '[] (no candidates above threshold)'}
Agent 3 verdict: {"NON-EMPTY — rules found confident candidates" if not _rule_empty else "EMPTY — rules found no candidates above threshold"}
Note: Agent 3 uses optimised parameters from 5×2 cross-validation on 80 training sequences.
      Its predictions for RELIABLE signs (A1, Z1, A40, B1, A24) are highly accurate.
      Missing signs not in KB: novel/variant signs absent from Agent 3 are not errors.
══════════════════════════════════════════════════════════

TASK — apply the THREE-AGENT VOTING RULES from the system prompt:
1) For each candidate sign: count how many agents support it (0/1/2/3).
2) Apply voting rules 1-5 in order to decide MAIN vs ALTERNATIVE vs REJECT.
3) In your justification: explicitly state the voting rule that applied and
   which agents agreed (e.g., "Rule 2 applied: Rules + Structural agree on A1").
4) ALTERNATIVES: signs supported by only 1 agent that might be promoted.
5) AMBIGUITY: state the overall agent agreement level for this sequence.

IMPORTANT:
- Every determinative must match tokens[index] exactly and must not be a separator.
- Include repeated occurrences separately (same sign at different indices).
- Do NOT include weak hypotheses as MAIN; place them under alternatives if needed.

Return JSON ONLY, strictly matching the schema.
"""

        # Invocar LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = llm.invoke(messages)
        synth_data = parse_llm_json(response.content if hasattr(response, 'content') else str(response))
        state['synthesizer_json'] = synth_data if synth_data else {}

        # Deterministic post-processing to reduce false positives and promote multi-source candidates
        state['synthesizer_json'] = postprocess_synth_json_with_scoring(state['synthesizer_json'], state)

        # Source of truth: synthesizer JSON (end-to-end)
        state['consolidated_analysis'] = format_report_from_synth_json(state['synthesizer_json'])

        main_obj = (state['synthesizer_json'].get('main_interpretation', {}) or {})
        main_det = main_obj.get('determinatives', None)
        if not isinstance(main_det, list):
            main_sem = main_obj.get('semantic_classifiers', []) or []
            main_fun = main_obj.get('functional_markers', []) or []
            main_det = list(main_sem) + list(main_fun)
        else:
            # keep as provided
            main_det = main_det or []
        state['determinatives'] = normalize_determinatives_from_json(
            main_det, 
            state['tokens'], 
            default_method='synth_json',
            determinatives_dict=state.get('enriched_determinatives'),
            state=state
        )

        
        print(f"✅ Síntese concluída: {len(state['determinatives'])} determinativos identificados")
        print(f"Determinativos encontrados:{state['determinatives']}")
        
    except Exception as e:
        print(f"⚠️  Erro no agente sintetizador: {e}")
        state['consolidated_analysis'] = f"Error: {str(e)}"
        state['determinatives'] = []
    
    return state


# ============================================================================
# 🌐 AGENTE DE TRADUÇÃO (ESPANHOL → INGLÊS)
# ============================================================================

def translate_analysis_to_english(spanish_analysis: str) -> str:
    """
    Traduz análise completa de espanhol para inglês
    
    Args:
        spanish_analysis: Texto completo da análise em espanhol
        
    Returns:
        Tradução fiel para inglês
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Usar o mesmo LLM configurado
        if os.getenv('ANTHROPIC_API_KEY'):
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.0)  # v1.42: temp=0.0
        else:
            llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0.0, top_p=1, seed=GLOBAL_SEED)
        
        system_prompt = """You are a FAITHFUL translator specializing in Egyptology.

YOUR TASK:
Translate the Egyptian hieroglyphic analysis from Spanish to English.

CRITICAL RULES:
1. FAITHFUL TRANSLATION ONLY - Do NOT reinterpret the content
2. PRESERVE all technical terminology (sign codes like A1, Z3, etc.)
3. MAINTAIN the original formatting (headers, bullets, separators)
4. Keep determinative lists EXACTLY as they are
5. Translate descriptions but keep hieroglyphic codes unchanged

WHAT TO TRANSLATE:
Headers and section titles
Descriptions and explanations
Justifications and reasoning  
General text

WHAT NOT TO CHANGE:
Sign codes (A1, Z3, D54, etc.)
MDC sequences
Determinative lists in brackets [ ]
Formatting symbols (===, ---, etc.)
Position numbers

EXAMPLE:
Spanish: "- Determinativos: [A1 (persona masculina), Z3 (plural)]"
English: "- Determinatives: [A1 (masculine person), Z3 (plural)]"

Spanish: "Justificación: A1 representa persona en posición 5"
English: "Justification: A1 represents person at position 5"

Remember: TRANSLATE, don't REINTERPRET!"""

        human_prompt = f"Translate this analysis faithfully to English:\n\n{spanish_analysis}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = llm.invoke(messages)
        
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
            
    except Exception as e:
        print(f"⚠️  Erro na tradução: {e}")
        return f"[Translation error: {e}]\n\n{spanish_analysis}"



# ============================================================================
# 🔍 AGENTE DE BUSCA DE SÍMBOLOS GARDINER
# ============================================================================

def gardiner_lookup_agent(sign_code: str) -> dict:
    """Busca definição de símbolo Gardiner usando LLM"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import SystemMessage, HumanMessage
        import os
        
        if os.getenv('ANTHROPIC_API_KEY'):
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.0)  # v1.42: temp=0.0
        else:
            llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0.0, top_p=1,seed=GLOBAL_SEED)
        
        system_prompt = """You are an expert in Egyptian hieroglyphics and the Gardiner sign list.

YOUR TASK: Given a Gardiner sign code, provide its definition and determinative role.

RESPONSE FORMAT (JSON only):
{
  "sign": "CODE",
  "gardiner_category": "Category name",
  "description": "Brief description",
  "is_determinative": true/false,
  "determinative_role": "Explanation if applicable",
  "semantic_field": "General field"
}

Return ONLY the JSON, no markdown."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Provide information for Gardiner sign: {sign_code}")
        ]
        
        response = llm.invoke(messages)
        content_text = response.content if hasattr(response, 'content') else str(response)
        
        import json
        json_match = re.search(r'\{[^}]+\}', content_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            result.setdefault('sign', sign_code)
            result.setdefault('gardiner_category', 'Unknown')
            result.setdefault('description', f'Symbol {sign_code}')
            result.setdefault('is_determinative', False)
            result.setdefault('determinative_role', 'Unknown')
            result.setdefault('semantic_field', 'unknown')
            return result
        else:
            return {'sign': sign_code, 'gardiner_category': 'Unknown', 'description': f'Symbol {sign_code}', 'is_determinative': False, 'determinative_role': 'Unknown', 'semantic_field': 'unknown'}
    except Exception as e:
        print(f"⚠️  Erro busca Gardiner {sign_code}: {e}")
        return {'sign': sign_code, 'gardiner_category': 'Unknown', 'description': f'Symbol {sign_code}', 'is_determinative': False, 'determinative_role': 'Unknown', 'semantic_field': 'unknown'}


def enrich_determinatives_with_lookup(tokens: list, determinatives_detailed: dict) -> dict:
    """Enriquece lista com busca de símbolos desconhecidos"""
    enriched_dict = determinatives_detailed.copy()
    unknown_signs = [token for token in tokens if token not in enriched_dict and token not in ['S29']]
    seen = set()
    unknown_signs = [x for x in unknown_signs if not (x in seen or seen.add(x))]
    
    if unknown_signs:
        print(f"\n🔍 Buscando {len(unknown_signs)} símbolos: {', '.join(unknown_signs)}")
        for sign in unknown_signs:
            info = gardiner_lookup_agent(sign)
            if info['is_determinative']:
                enriched_dict[sign] = {
                    'category': info['gardiner_category'],
                    'meaning': info['description'],
                    'determinative_role': info['determinative_role'],
                    'semantic_field': info['semantic_field']
                }
                print(f"   ✅ {sign}: {info['description']}")
    
    return enriched_dict


# ============================================================================
# CRIAÇÃO DO GRAFO LANGGRAPH
# ============================================================================

def create_langgraph_workflow():
    """
    Cria o workflow LangGraph que conecta os três agentes
    """
    if not USE_LANGGRAPH:
        return None
    
    try:
        from langgraph.graph import StateGraph, END
        
        # Criar grafo
        workflow = StateGraph(AgentState)
        
        # ====================================================================
        # V1.15 - ESTRUTURAL INDEPENDENTE (preparado para paralelização)
        # ====================================================================
        # NOTA: LangGraph v0.x não suporta múltiplos entry points diretamente
        # Mantemos fluxo linear MAS estrutural é independente do semântico
        # Isso facilita paralelização futura quando LangGraph suportar nativamente
        # ====================================================================
        
        # Adicionar nodos (agentes)
        workflow.add_node("word_boundary_agent", word_boundary_agent_node)
        workflow.add_node("semantic_agent", semantic_agent_node)
        workflow.add_node("structural_agent", structural_agent_node)
        workflow.add_node("synthesizer_agent", synthesizer_agent_node)
        
        # Definir fluxo LINEAR (por limitação do LangGraph)
        # 1. Começa no agente semântico
        workflow.set_entry_point("word_boundary_agent")
        
        # 1b. Depois vai para o agente semântico
        workflow.add_edge("word_boundary_agent", "semantic_agent")

        # 2. Depois vai para o agente estrutural (MAS não usa resultado do semântico!)
        workflow.add_edge("semantic_agent", "structural_agent")
        
        # 3. Depois vai para o sintetizador
        workflow.add_edge("structural_agent", "synthesizer_agent")
        
        # 4. Termina após o sintetizador
        workflow.add_edge("synthesizer_agent", END)
        
        # Compilar grafo
        app = workflow.compile()
        
        return app
        
    except Exception as e:
        print(f"⚠️  Erro ao criar LangGraph workflow: {e}")
        return None

# Inicializar workflow global
LANGGRAPH_APP = create_langgraph_workflow()

# ============================================================================
# FUNÇÃO PRINCIPAL DE ANÁLISE COM LANGGRAPH
# ============================================================================

def analyze_with_langgraph(mdc_sequence: str, image_file: str = "",
                           expert_examples: list = None,
                           optimised_params: Dict[str, float] = None):
    """
    Analisa sequência MdC usando o sistema multi-agente LangGraph
    
    Args:
        mdc_sequence: sequência em formato Manuel de Codage
        image_file: nome do arquivo de imagem (opcional)
    
    Returns:
        dict com análises e determinativos identificados
    """
    if not USE_LANGGRAPH or LANGGRAPH_APP is None:
        # Fallback para análise por regras
        tokens = tokenize_mdc(mdc_sequence)
        rule_result = analyze_rule_based(tokens)
        return {
            'success': False,
            'error': 'LangGraph não disponível',
            'mdc_sequence': mdc_sequence,
            'image_file': image_file,
            'tokens': tokens,
            'determinatives': rule_result['determinatives'],
            'method': 'rule_based'
        }
    
    try:
        # Tokenizar sequência
        tokens = tokenize_mdc(mdc_sequence)
        
        # Análise por regras (baseline)
        rule_result = analyze_rule_based(tokens)
        
        # Preparar informação de candidatos
        candidates_info = []
        for token in tokens:
            if token in DETERMINATIVES_DETAILED:
                info = DETERMINATIVES_DETAILED[token]
                candidates_info.append(
                    f"  - {token}: {info['significado']} "
                    f"(cat: {info['categoria']}, sem: {info['semantica']})"
                )
        
        candidates_text = "\n".join(candidates_info) if candidates_info else "  (ningún candidato en la base)"

        # ── V1.33 EMPTY-SEQUENCE GUARD (soft version) ─────────────────────────
        # If the KB finds no candidates, the rule system predicts empty.
        # However, some sequences contain out-of-KB signs that ARE determinatives
        # according to expert annotation. A hard block would cause FN_NULL on these.
        # Solution: let the LLM run but pass a strong conservative warning so it
        # only predicts when confidence is very high. This prevents hallucination
        # while still catching legitimate out-of-KB determinatives.
        _no_kb_candidates = not candidates_info
        _kb_warning = ""
        if _no_kb_candidates:
            print(f"  ⚠️  No KB candidates — LLM will run with conservative warning")
            _kb_warning = (
                "\n\n⚠️  CRITICAL WARNING: None of the signs in this sequence appear "
                "in the knowledge base as known determinatives. The rule-based system "
                "predicts NO determinatives for this sequence. "
                "You MUST be extremely conservative: "
                "ONLY output a determinative if your confidence exceeds 0.90. "
                "If there is ANY doubt, output an empty list []."
            )
        _examples_to_use = expert_examples if expert_examples is not None else EXPERT_EXAMPLES
        expert_context = format_expert_examples(_examples_to_use) if _examples_to_use else ""

        # V1.35: build agent-specific guidance from optimised params
        # Each agent receives ONLY the guidance relevant to its reasoning role.
        # The Sintetizador previously received NO guidance — this is now fixed.
        semantic_guidance    = format_params_for_semanticist(optimised_params)    if optimised_params else ""
        structural_guidance  = format_params_for_structuralist(optimised_params)  if optimised_params else ""
        synthesizer_guidance = format_params_for_synthesizer(optimised_params)    if optimised_params else ""

        enriched_determinatives = DETERMINATIVES_DETAILED

        # Preparar estado inicial
        initial_state = {
            'mdc_sequence': mdc_sequence,
            'image_file': image_file,
            'tokens': tokens,
            'candidates_info': candidates_info,
            'candidates_text': candidates_text + _kb_warning,
            'rule_result': rule_result,
            'expert_context': expert_context,       # few-shot examples (shared by all)
            'semantic_guidance':    semantic_guidance,    # V1.35: Semanticista only
            'structural_guidance':  structural_guidance,  # V1.35: Estruturalista only
            'synthesizer_guidance': synthesizer_guidance, # V1.35: Sintetizador only (was empty before!)
            'enriched_determinatives': enriched_determinatives,
            'mdc_structure': build_mdc_structure(tokens),
            'words': [],
            'semantic_analysis': "",
            'structural_analysis': "",
            'consolidated_analysis': "",
            'determinatives': [],
            'semantic_json': {},
            'structural_json': {},
            'synthesizer_json': {},
            'method': 'multi_agent_langgraph',
            'success': True,
            'error': ""
        }
        
        # Executar workflow LangGraph
        print(f"\n🔄 Processando: {mdc_sequence[:50]}...")
        final_state = LANGGRAPH_APP.invoke(initial_state)
        
        return {
            'success': True,
            'mdc_sequence': mdc_sequence,
            'image_file': image_file,
            'tokens': tokens,
            'semantic_analysis': final_state['semantic_analysis'],
            'structural_analysis': final_state['structural_analysis'],
            'consolidated_analysis': final_state['consolidated_analysis'],
            'determinatives': final_state['determinatives'],
            'has_alternatives': True,
            'method': 'multi_agent_langgraph'
        }
        
    except Exception as e:
        print(f"⚠️  Erro na análise LangGraph: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'mdc_sequence': mdc_sequence,
            'image_file': image_file,
            'method': 'error'
        }

# ============================================================================
# PROCESSAMENTO DO DATASET
# ============================================================================

print("\n" + "="*70)
print("🔄 PROCESSANDO DATASET COM LANGGRAPH")
print("="*70)
print(f"Total de sequências: {len(df)}")

# Adicionar colunas para resultados
df['determinative_count'] = 0
df['determinatives_found'] = None
df['analysis_method'] = 'rule_based'
df['langgraph_analysis'] = None

# ============================================================================
# V1.32: DESPACHO DO MODO DE AVALIAÇÃO
# ============================================================================
#   EVAL_MODE = 'nested_cv' → 5 × (5×2 CV interno + Teste LLM)  ← v1.32
#   EVAL_MODE = 'loo'       → Leave-One-Out completo              ← v1.29-v1.31
#   EVAL_MODE = 'normal'    → processamento padrão sem avaliação  ← v1.28
# ============================================================================

nested_cv_result = None   # preenchido se EVAL_MODE='nested_cv'
loo_results      = []     # preenchido se EVAL_MODE='loo'

if EVAL_MODE == 'nested_cv' and LOO_DF is not None and LOO_GT_DICT:
    # ──────────────────────────────────────────────────────────────────────
    # NESTED CV: 5 × (5×2 CV + Teste LLM com 19 amostras)
    # ──────────────────────────────────────────────────────────────────────
    nested_cv_result = run_nested_cv(LOO_DF, LOO_GT_DICT)
    print_nested_cv_report(nested_cv_result)
    export_nested_cv_results(nested_cv_result, OUTPUT_DIR)

    # ── V1.36: Strategy B — Consensus Parameter Evaluation ───────────────────
    if RUN_CONSENSUS_CV:
        print("\n" + "="*70)
        print("V1.36: Running Strategy B (consensus params + v1.35 guidance)...")
        print("="*70)
        _outer_splits_for_consensus = _make_outer_splits(
            LOO_DF, OUTER_N_SPLITS, OUTER_RANDOM_STATE)
        _consensus_result = _run_consensus_cv(
            LOO_DF, LOO_GT_DICT, nested_cv_result, _outer_splits_for_consensus)
        print_strategy_comparison(_consensus_result)
        _plot_strategy_comparison(_consensus_result, OUTPUT_DIR)
        _export_consensus_results(_consensus_result, OUTPUT_DIR)

elif EVAL_MODE == 'loo' and LOO_DF is not None:
    # ──────────────────────────────────────────────────────────────────────
    # MODO LOO (v1.29-v1.31): processar cada uma das 99 amostras
    print(f"\n🔬 V1.29 LOO: Processando {len(LOO_DF)} amostras com avaliação Leave-One-Out")
    if LOO_MAX_SAMPLES is not None:
        print(f"   (limitado a {LOO_MAX_SAMPLES} amostras para este run)")

    loo_results = []  # acumula resultados para avaliação final

    _df_to_process_loo = LOO_DF if LOO_MAX_SAMPLES is None else LOO_DF.iloc[:LOO_MAX_SAMPLES]

    for loo_i, (_, loo_row) in enumerate(_df_to_process_loo.iterrows()):
        mdc_sequence = loo_row['Manuel_de_Codage'] if 'Manuel_de_Codage' in loo_row else loo_row['Code']
        image_file   = loo_row.get('image_file', f'img_loo_{loo_i}')

        if pd.isna(mdc_sequence) or str(mdc_sequence).strip() == '':
            continue

        # ── LOO: exemplos few-shot excluem a amostra atual ───────────────
        # Identifica a amostra atual no LOO_DF pelo índice posicional
        _df_other = LOO_DF.drop(LOO_DF.index[loo_i]).reset_index(drop=True)
        loo_examples = select_expert_examples_loo(_df_other, n_examples=LOO_N_FEW_SHOT)

        print(f"\n[LOO {loo_i+1}/{len(_df_to_process_loo)}] {str(mdc_sequence)[:55]}")
        print(f"  Few-shot examples (from {len(_df_other)} other): {len(loo_examples)}")

        # ── Análise ──────────────────────────────────────────────────────
        if USE_LANGGRAPH:
            result = analyze_with_langgraph(str(mdc_sequence), str(image_file),
                                            expert_examples=loo_examples)
        else:
            tokens = tokenize_mdc(str(mdc_sequence))
            rule_result = analyze_rule_based(tokens)
            result = {
                'success': True,
                'mdc_sequence': str(mdc_sequence),
                'tokens': tokens,
                'determinatives': rule_result['determinatives'],
                'method': 'rule_based',
                'consolidated_analysis': ''
            }

        # ── Guardar resultado no df principal ────────────────────────────
        # Encontrar a linha correspondente no df original
        _norm_seq = normalize_mdc(str(mdc_sequence))
        _orig_mask = df['Code'].apply(lambda c: normalize_mdc(str(c)) if isinstance(c, str) else '') == _norm_seq
        _orig_idx  = df[_orig_mask].index

        if result.get('success', False) or result.get('determinatives'):
            _dets = result.get('determinatives', [])
            if len(_orig_idx) > 0:
                df.at[_orig_idx[0], 'determinative_count'] = len(_dets)
                df.at[_orig_idx[0], 'determinatives_found'] = _dets
                df.at[_orig_idx[0], 'analysis_method'] = result.get('method', 'multi_agent')
                df.at[_orig_idx[0], 'langgraph_analysis'] = result.get('consolidated_analysis', '')
        else:
            tokens = tokenize_mdc(str(mdc_sequence))
            rule_result = analyze_rule_based(tokens)
            _dets = rule_result['determinatives']
            if len(_orig_idx) > 0:
                df.at[_orig_idx[0], 'determinative_count'] = len(_dets)
                df.at[_orig_idx[0], 'determinatives_found'] = _dets
                df.at[_orig_idx[0], 'analysis_method'] = 'rule_based'

        # ── Comparar com GT para métricas LOO ────────────────────────────
        predicted_signs = [d['sign'] for d in _dets if isinstance(d, dict) and d.get('sign')]
        _norm_seq_gt = re.sub(r'\s+', '', str(mdc_sequence).strip())
        gt_signs = LOO_GT_DICT.get(_norm_seq_gt, [])
        sample_metrics = evaluate_single_prediction(predicted_signs, gt_signs)

        loo_results.append({
            'loo_index':       loo_i,
            'sequence':        str(mdc_sequence),
            'predicted_signs': predicted_signs,
            'gt_signs':        gt_signs,
            'metrics':         sample_metrics,
            'few_shot_n':      len(loo_examples),
        })

        print(f"  Pred: {predicted_signs}  GT: {gt_signs}")
        print(f"  P={sample_metrics['precision']*100:.1f}%  "
              f"R={sample_metrics['recall']*100:.1f}%  "
              f"F1={sample_metrics['f1']*100:.1f}%")

    print(f"\n✅ LOO processamento concluído! {len(loo_results)} amostras avaliadas.")

else:
    # ──────────────────────────────────────────────────────────────────────
    # MODO NORMAL (v1.28): processar quantidade/range configurado de frases
    # (também ativado quando EVAL_MODE='normal' ou GT indisponível)
    # ──────────────────────────────────────────────────────────────────────

    if INICIO_RANGE is not None or FIM_RANGE is not None:
        inicio = INICIO_RANGE if INICIO_RANGE is not None else 0
        fim    = FIM_RANGE    if FIM_RANGE    is not None else len(df)
        df_to_process = df.iloc[inicio:fim]
        print(f"📊 Processando RANGE de frases: {inicio} a {fim}")
        print(f"   Total: {len(df_to_process)} frases (de {len(df)} totais)")
    elif NUM_FRASES_PROCESSAR is None:
        df_to_process = df
        print(f"📊 Processando TODAS as {len(df)} frases do dataset")
    else:
        df_to_process = df.head(NUM_FRASES_PROCESSAR)
        print(f"📊 Processando {len(df_to_process)} primeiras frases (de {len(df)} totais)")

    for idx, row in df_to_process.iterrows():
        mdc_sequence = row['Manuel_de_Codage']
        image_file   = row.get('image_file', f'img_{idx}')

        if pd.isna(mdc_sequence) or mdc_sequence == '':
            continue

        if USE_LANGGRAPH:
            result = analyze_with_langgraph(mdc_sequence, image_file)
            if result['success']:
                df.at[idx, 'determinative_count'] = len(result['determinatives'])
                df.at[idx, 'determinatives_found'] = result['determinatives']
                df.at[idx, 'analysis_method'] = 'multi_agent'
                df.at[idx, 'langgraph_analysis'] = result['consolidated_analysis']
            else:
                tokens = tokenize_mdc(mdc_sequence)
                rule_result = analyze_rule_based(tokens)
                df.at[idx, 'determinative_count'] = len(rule_result['determinatives'])
                df.at[idx, 'determinatives_found'] = rule_result['determinatives']
                df.at[idx, 'analysis_method'] = 'rule_based'
        else:
            tokens = tokenize_mdc(mdc_sequence)
            rule_result = analyze_rule_based(tokens)
            df.at[idx, 'determinative_count'] = len(rule_result['determinatives'])
            df.at[idx, 'determinatives_found'] = rule_result['determinatives']
            df.at[idx, 'analysis_method'] = 'rule_based'

        if (idx + 1) % 50 == 0:
            print(f"  Processadas: {idx + 1}/{len(df)}")

print("\n✅ Processamento concluído!")
print(f"   Sequências processadas (total df): {len(df)}")
print(f"   Com LangGraph: {(df['analysis_method'] == 'multi_agent').sum()}")
print(f"   Apenas regras: {(df['analysis_method'] == 'rule_based').sum()}")

# ============================================================================
# V1.29: AVALIAÇÃO E RELATÓRIO LOO
# ============================================================================

if LOO_MODE and loo_results:
    # ── Calcular métricas globais ─────────────────────────────────────────
    loo_metrics = compute_loo_metrics(loo_results)
    print_loo_report(loo_metrics, loo_results)

    # ── Exportar CSV detalhado por amostra ───────────────────────────────
    loo_df_rows = []
    for r in loo_results:
        m = r['metrics']
        loo_df_rows.append({
            'loo_index':        r['loo_index'],
            'sequence':         r['sequence'],
            'outcome_type':     m.get('outcome_type', 'UNKNOWN'),
            'predicted_signs':  ','.join(r['predicted_signs']),
            'gt_signs':         ','.join(r['gt_signs']),
            'tp':               m['tp'],
            'fp':               m['fp'],
            'fn':               m['fn'],
            'precision':        round(m['precision'], 4),
            'recall':           round(m['recall'], 4),
            'f1':               round(m['f1'], 4),
            'null_correct':     m.get('null_correct', False),
            'fp_null':          m.get('fp_null', False),
            'fn_null':          m.get('fn_null', False),
            'few_shot_n':       r.get('few_shot_n', 0),
        })
    loo_eval_df = pd.DataFrame(loo_df_rows)
    loo_csv_path = f"{OUTPUT_DIR}/loo_evaluation_per_sample.csv"
    loo_eval_df.to_csv(loo_csv_path, index=False, encoding='utf-8')
    print(f"✅ LOO avaliação por amostra: {loo_csv_path}")

    # ── Exportar métricas por signo ───────────────────────────────────────
    sign_rows = []
    for s, sm in sorted(loo_metrics['per_sign'].items()):
        sign_rows.append({
            'sign': s,
            'tp': sm['tp'], 'fp': sm['fp'], 'fn': sm['fn'],
            'precision': round(sm['precision'], 4),
            'recall':    round(sm['recall'],    4),
            'f1':        round(sm['f1'],        4),
        })
    sign_df = pd.DataFrame(sign_rows).sort_values('f1', ascending=False)
    sign_csv_path = f"{OUTPUT_DIR}/loo_evaluation_per_sign.csv"
    sign_df.to_csv(sign_csv_path, index=False, encoding='utf-8')
    print(f"✅ LOO métricas por signo: {sign_csv_path}")

    # ── Exportar métricas globais em JSON ─────────────────────────────────
    import json as _json_loo
    global_summary = {
        'version': 'v1.31_LOO_null_aware_with_optimization',
        'n_samples': loo_metrics['n_samples'],
        'micro': {k: round(float(v), 4) for k, v in loo_metrics['micro'].items()
                  if isinstance(v, (int, float))},
        'macro': {k: round(float(v), 4) for k, v in loo_metrics['macro'].items()
                  if isinstance(v, (int, float))},
        'macro_classic': {k: round(float(v), 4)
                          for k, v in loo_metrics.get('macro_classic', {}).items()
                          if isinstance(v, (int, float))},
        'null_stats': {k: (round(float(v), 4) if isinstance(v, float) else v)
                       for k, v in loo_metrics.get('null_stats', {}).items()
                       if v is not None},
    }
    summary_path = f"{OUTPUT_DIR}/loo_global_metrics.json"
    with open(summary_path, 'w', encoding='utf-8') as _f:
        _json_loo.dump(global_summary, _f, indent=2, ensure_ascii=False)
    print(f"✅ LOO métricas globais (JSON): {summary_path}")

    # ── Gráfico de distribuição de F1 por amostra ────────────────────────
    f1_scores = [r['metrics']['f1'] for r in loo_results]
    plt.figure(figsize=(10, 5))
    plt.hist(f1_scores, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    plt.axvline(loo_metrics['micro']['f1'], color='red', linestyle='--',
                label=f"Micro-F1 = {loo_metrics['micro']['f1']*100:.1f}%")
    plt.axvline(loo_metrics['macro']['f1'], color='orange', linestyle='-.',
                label=f"Macro-F1 = {loo_metrics['macro']['f1']*100:.1f}%")
    plt.xlabel('F1-Score por Sequência', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.title('Distribuição de F1-Score — Avaliação LOO (v1.29)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    loo_hist_path = f"{OUTPUT_DIR}/loo_f1_distribution.png"
    plt.savefig(loo_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico LOO F1: {loo_hist_path}")

    # ── Gráfico P/R/F1 por signo (top-15) ────────────────────────────────
    _ps = loo_metrics['per_sign']
    _top_signs = sorted(_ps.items(), key=lambda x: -(x[1]['tp'] + x[1]['fn']))[:15]
    _sign_names = [s for s, _ in _top_signs]
    _precisions = [sm['precision'] * 100 for _, sm in _top_signs]
    _recalls    = [sm['recall']    * 100 for _, sm in _top_signs]
    _f1s        = [sm['f1']        * 100 for _, sm in _top_signs]
    _x = range(len(_sign_names))
    plt.figure(figsize=(13, 6))
    w = 0.28
    plt.bar([xi - w for xi in _x], _precisions, w, label='Precision', color='steelblue',  alpha=0.85)
    plt.bar([xi      for xi in _x], _recalls,    w, label='Recall',    color='seagreen',   alpha=0.85)
    plt.bar([xi + w  for xi in _x], _f1s,        w, label='F1',        color='darkorange',  alpha=0.85)
    plt.xticks(_x, _sign_names, rotation=45, ha='right')
    plt.ylabel('Score (%)', fontsize=11)
    plt.title('P / R / F1 por Signo — Top-15 (LOO v1.29)', fontsize=13, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    sign_chart_path = f"{OUTPUT_DIR}/loo_per_sign_metrics.png"
    plt.savefig(sign_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico LOO métricas por signo: {sign_chart_path}")

#%%
# ============================================================================
# VISUALIZAÇÃO DE RESULTADOS
# ============================================================================

def visualize_determinatives(df_with_det, output_dir):
    """
    Visualização dos resultados (idêntica ao código CrewAI)
    """
    print("\n" + "="*70)
    print("📊 VISUALIZAÇÃO DOS RESULTADOS")
    print("="*70 + "\n")
    
    # Extrair todos os determinativos
    all_determinatives = []
    for idx, row in df_with_det.iterrows():
        if row['determinative_count'] > 0 and row['determinatives_found']:
            all_determinatives.extend(row['determinatives_found'])
    
    # Estatísticas gerais
    total_det = len(all_determinatives)
    unique_signs = len(set(d['sign'] for d in all_determinatives))
    
    print(f"📊 Total de determinativos: {total_det}")
    print(f"🔤 Signos únicos: {unique_signs}")
    
    # Contar por método de detecção
    if USE_LANGGRAPH:
        lg_dets = [d for d in all_determinatives if d.get('method') == 'langgraph']
        rule_dets = [d for d in all_determinatives if d.get('method') == 'rules']
        print(f"🤖 Detectados por LangGraph: {len(lg_dets)}")
        print(f"📋 Detectados por regras: {len(rule_dets)}")
    
    # 1. Top 10 determinativos mais frequentes
    sign_counts = Counter(d['sign'] for d in all_determinatives)
    top_10 = sign_counts.most_common(10)

    print("\n🏆 TOP 10 DETERMINATIVOS MAIS FREQUENTES:")
    if not top_10:
        print("   (sem dados — em EVAL_MODE='nested_cv' os determinativos são")
        print("    avaliados por fold e não acumulados no df principal)")
    else:
        for i, (sign, count) in enumerate(top_10, 1):
            info = DETERMINATIVES_DETAILED.get(sign, {})
            significado = info.get('significado', 'desconocido')
            print(f"   {i}. {sign}: {count}x - {significado}")

    # Gráfico de barras
    if top_10:
        plt.figure(figsize=(12, 6))
        signs, counts = zip(*top_10)
        plt.bar(signs, counts, color='steelblue', alpha=0.8)
        plt.xlabel('Signo de Determinativo', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.title('Top 10 Determinativos Más Frecuentes' +
                  (' (Sistema Multi-Agente LangGraph)' if USE_LANGGRAPH else ' (Reglas)'),
                  fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top10_determinatives.png", dpi=300, bbox_inches='tight')
        print(f"\n💾 Gráfico guardado: {output_dir}/top10_determinatives.png")
        plt.close()

    # 2. Distribuição por categoria
    category_counts = Counter(
        DETERMINATIVES_DETAILED.get(d['sign'], {}).get('categoria', 'Desconocido')
        for d in all_determinatives)

    print("\n📂 DISTRIBUIÇÃO POR CATEGORIA:")
    if not category_counts:
        print("   (sem dados)")
    else:
        for cat, count in category_counts.most_common():
            pct = (count / total_det) * 100
            print(f"   {cat}: {count} ({pct:.1f}%)")

    # Gráfico de pizza
    if not category_counts:
        return
    categories, counts = zip(*category_counts.most_common())
    plt.figure(figsize=(10, 8))
    plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
    plt.title('Distribución de Determinativos por Categoría Semántica' +
              ('\n(Sistema Multi-Agente LangGraph)' if USE_LANGGRAPH else '\n(Análisis por Reglas)'),
              fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/categories_distribution.png", dpi=300, bbox_inches='tight')
    print(f"💾 Gráfico guardado: {output_dir}/categories_distribution.png")
    plt.close()
    
    # 3. Distribuição de confiança (se disponível)
    if any('confidence' in d for d in all_determinatives):
        confidence_counts = Counter(d.get('confidence', 'unknown') for d in all_determinatives)
        
        print("\n🎯 DISTRIBUIÇÃO DE CONFIANÇA:")
        for conf, count in confidence_counts.most_common():
            pct = (count / total_det) * 100
            print(f"   {conf}: {count} ({pct:.1f}%)")
    
    # 4. Análise LangGraph (se disponível)
    if USE_LANGGRAPH:
        lg_analyzed = df_with_det[df_with_det['analysis_method'] == 'multi_agent']
        if len(lg_analyzed) > 0:
            print("\n🤖 ANÁLISES DO SISTEMA LANGGRAPH:")
            print(f"   Total: {len(lg_analyzed)} sequências")
            
            # Mostrar 3 exemplos
            print("\n📝 EXEMPLOS DE ANÁLISE DO SISTEMA:\n")
            for idx, row in lg_analyzed.head(3).iterrows():
                print(f"{'='*70}")
                print(f"Sequência: {row['Manuel_de_Codage']}")
                print(f"Determinativos: {[d['sign'] for d in row['determinatives_found']]}")
                print("\nAnálise do Sistema:")
                analysis = row['langgraph_analysis']
                if analysis:
                    # Limitar a 500 caracteres para não sobrecarregar output
                    print(analysis[:500] + ("..." if len(analysis) > 500 else ""))
                print()
    
    print("="*70)
    print("✅ VISUALIZAÇÃO CONCLUÍDA")
    print("="*70 + "\n")

# Executar visualização
visualize_determinatives(df, OUTPUT_DIR)

# ============================================================================
# EXPORTAR RESULTADOS
# ============================================================================

print("\n💾 Exportando resultados...")

# 1. CSV com determinativos detalhados
export_data = []

if EVAL_MODE == 'nested_cv' and nested_cv_result is not None:
    # Build from nested CV test results — these are the actual LLM predictions
    for outer_r in nested_cv_result['outer_results']:
        for sr in outer_r['llm_test_results']:
            seq = sr['sequence']
            for sign_code in sr['predicted_signs']:
                det_info = DETERMINATIVES_DETAILED.get(sign_code, {})
                export_data.append({
                    'outer_fold':          outer_r['outer_fold'] + 1,
                    'mdc_sequence':        seq,
                    'determinative_sign':  sign_code,
                    'gt_signs':            ','.join(sr['gt_signs']),
                    'outcome_type':        sr['metrics'].get('outcome_type', ''),
                    'significado':         det_info.get('significado', 'desconocido'),
                    'categoria':           det_info.get('categoria', 'Desconocido'),
                    'semantica':           det_info.get('semantica', 'N/A'),
                    'f1':                  round(sr['metrics']['f1'], 4),
                    'precision':           round(sr['metrics']['precision'], 4),
                    'recall':              round(sr['metrics']['recall'], 4),
                })
else:
    # Normal / LOO mode — read from df as before
    for idx, row in df.iterrows():
        if row['determinative_count'] > 0:
            for det in row['determinatives_found']:
                sign_code = det['sign']
                det_info  = DETERMINATIVES_DETAILED.get(sign_code, {})
                export_data.append({
                    'index':               idx,
                    'image_file':          row.get('image_file', ''),
                    'mdc_sequence':        row['Manuel_de_Codage'],
                    'determinative_sign':  sign_code,
                    'position':            det['position'],
                    'significado':         det_info.get('significado', 'desconocido'),
                    'categoria':           det_info.get('categoria', 'Desconocido'),
                    'semantica':           det_info.get('semantica', 'N/A'),
                    'confidence':          det.get('confidence', 'unknown'),
                    'method':              det.get('method', 'unknown'),
                    'expert_translation':  row.get('expert_translation', ''),
                })

df_export = pd.DataFrame(export_data)
csv_path = f"{OUTPUT_DIR}/determinatives_detailed.csv"
df_export.to_csv(csv_path, index=False, encoding='utf-8')
print(f"✅ CSV exportado: {csv_path} ({len(df_export)} linhas)")

# 2. Dataset completo com colunas de determinativos
full_export_path = f"{OUTPUT_DIR}/dataset_with_determinatives.csv"
if EVAL_MODE == 'nested_cv' and nested_cv_result is not None:
    # Flatten all test results into one row per sequence per outer fold
    rows = []
    for outer_r in nested_cv_result['outer_results']:
        for sr in outer_r['llm_test_results']:
            mm = sr['metrics']
            rows.append({
                'outer_fold':       outer_r['outer_fold'] + 1,
                'mdc_sequence':     sr['sequence'],
                'predicted_signs':  ','.join(sr['predicted_signs']),
                'gt_signs':         ','.join(sr['gt_signs']),
                'outcome_type':     mm.get('outcome_type', ''),
                'tp': mm['tp'], 'fp': mm['fp'], 'fn': mm['fn'],
                'precision':        round(mm['precision'], 4),
                'recall':           round(mm['recall'], 4),
                'f1':               round(mm['f1'], 4),
            })
    pd.DataFrame(rows).to_csv(full_export_path, index=False, encoding='utf-8')
else:
    df.to_csv(full_export_path, index=False, encoding='utf-8')
print(f"✅ Dataset completo: {full_export_path}")

#%%
# 3. Análises LangGraph (se disponível)
if USE_LANGGRAPH:
    lg_analyses = []
    for idx, row in df.iterrows():
        if row['analysis_method'] == 'multi_agent' and row['langgraph_analysis']:
            lg_analyses.append({
                'index': idx,
                'image_file': row.get('image_file', ''),
                'mdc_sequence': row['Manuel_de_Codage'],
                'analysis': row['langgraph_analysis'],
                'determinatives': row.get('determinatives_found', []),
                'expert_translation': row.get('expert_translation', '')
            })
    
    if lg_analyses:
        # Exportar como JSON
        json_path = f"{OUTPUT_DIR}/langgraph_analyses.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(lg_analyses, f, ensure_ascii=False, indent=2)
        print(f"✅ Análises LangGraph (JSON): {json_path}")
        
        # Exportar como TXT legível
        txt_path = f"{OUTPUT_DIR}/langgraph_analyses_ES.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for analysis in lg_analyses:
                f.write("="*70 + "\n")
                f.write(f"Archivo de imagen: {analysis['image_file']}\n")
                f.write(f"Sequência: {analysis['mdc_sequence']}\n")
                if 'expert_translation' in analysis and analysis['expert_translation']:
                    f.write(f"Especialista: {analysis['expert_translation']}\n")
                f.write("-"*70 + "\n")
                f.write(analysis['analysis'])
                f.write("\n\n")
        print(f"✅ Análises LangGraph (TXT): {txt_path}")
        
        # Gerar versão em INGLÊS
        print("\n🌐 Gerando versão em inglês...")
        txt_path_en =  f"{OUTPUT_DIR}/langgraph_analyses_EN.txt"
        
        with open(txt_path_en, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("EGYPTIAN HIEROGLYPHIC DETERMINATIVE ANALYSIS - ENGLISH VERSION\n")
            f.write("="*70 + "\n\n")
            
            for analysis in lg_analyses:
                # analysis já é um dicionário
                
                # Traduzir análise completa
                spanish_analysis = analysis['analysis']
                english_analysis = translate_analysis_to_english(spanish_analysis)
                
                # Escrever entrada
                f.write("="*70 + "\n")
                f.write(f"Image file: {analysis['image_file']}\n")
                f.write(f"Sequence: {analysis['mdc_sequence']}\n")
                
                if 'expert_translation' in analysis and analysis['expert_translation']:
                    f.write(f"Expert: {analysis['expert_translation']}\n")
                f.write("-"*70 + "\n")
                f.write(english_analysis)
                f.write("\n\n")
        
        print(f"✅ English version (TXT): {txt_path_en}")

print(f"\n✅ Todos os arquivos exportados para: {OUTPUT_DIR}")

#%%

# ============================================================================
# RESUMO FINAL
# ============================================================================

print("\n" + "="*70)
print("🎉 ANÁLISE COMPLETA!")
print("="*70 + "\n")

print("📊 RESUMO:")

if EVAL_MODE == 'nested_cv' and nested_cv_result is not None:
    # In nested_cv mode the main df loop is skipped — count determinatives
    # from the actual test results collected across all outer folds
    _all_test_results = [
        sr for r in nested_cv_result['outer_results']
        for sr in r['llm_test_results']
    ]
    _total_dets = sum(len(sr['predicted_signs']) for sr in _all_test_results)
    _total_seqs = len(_all_test_results)
    print(f"   • Sequências avaliadas (teste LLM): {_total_seqs}  "
          f"(de {len(df)} no dataset completo)")
    print(f"   • Determinativos encontrados (teste): {_total_dets}")
    print(f"   • Média por sequência: "
          f"{_total_dets/_total_seqs:.2f}" if _total_seqs else "   • N/A")
else:
    print(f"   • Sequências processadas: {len(df)}")
    print(f"   • Determinativos: {df['determinative_count'].sum()}")

print(f"   • Método usado: {'LangGraph Multi-Agente' if USE_LANGGRAPH else 'Apenas Regras'}")
print(f"   • Arquivos gerados: {OUTPUT_DIR}")

if nested_cv_result is not None:
    agg = nested_cv_result['aggregate']
    print(f"\n📊 NESTED CV (v1.32):")
    print(f"   • Folds externos: {OUTER_N_SPLITS}  |  Inner reps: {INNER_N_REPS}×2")
    if 'llm_macro_f1' in agg and 'mean' in agg['llm_macro_f1']:
        _lf = agg['llm_macro_f1']
        _vf = agg['val_f1_mean']
        print(f"   • Val  Macro-F1 : {_vf['mean']*100:.2f}% ± {_vf['std']*100:.2f}%")
        print(f"   • Test Macro-F1 : {_lf['mean']*100:.2f}% ± {_lf['std']*100:.2f}%")
        if 'generalization_gap' in agg and 'mean' in agg['generalization_gap']:
            _gap = agg['generalization_gap']
            print(f"   • Gap (test-val): {_gap['mean']*100:+.2f}% ± {_gap['std']*100:.2f}%")

elif loo_results:
    print(f"\n🔬 AVALIAÇÃO LOO (v1.31):")
    _m = loo_metrics['micro']
    _ma = loo_metrics['macro']
    print(f"   • Amostras avaliadas: {loo_metrics['n_samples']}")
    print(f"   • Micro  — F1={_m['f1']*100:.2f}%")
    print(f"   • Macro  — F1={_ma['f1']*100:.2f}%")

print("\n📁 ARQUIVOS DISPONÍVEIS:")
print("   • determinatives_detailed.csv - Tabela detalhada")
print("   • dataset_with_determinatives.csv - Dataset completo")
print("   • top10_determinatives.png - Gráfico de frequência")
print("   • categories_distribution.png - Distribuição por categoria")
if nested_cv_result is not None:
    print("   • nested_cv_per_fold.csv        - Métricas por fold externo")
    print("   • nested_cv_test_samples.csv    - Resultados por amostra de teste")
    print("   • nested_cv_inner_records.csv   - Registos inner 5×2 CV")
    print("   • nested_cv_global_metrics.json - Sumário agregado")
    print("   • nested_cv_f1_per_fold.png     - Gráfico comparativo por fold")
elif loo_results:
    print("   • loo_evaluation_per_sample.csv - F1/P/R por amostra LOO")
    print("   • loo_evaluation_per_sign.csv   - F1/P/R por signo LOO")
    print("   • loo_global_metrics.json       - Métricas globais LOO")
    print("   • loo_f1_distribution.png       - Histograma de F1 LOO")
    print("   • loo_per_sign_metrics.png      - Gráfico P/R/F1 por signo LOO")
if OPTIMIZE_PARAMS and '_best_params' in dir():
    print("   • optimized_params_v1_30.json   - Melhores parâmetros (cache)")
    print("   • optimized_params_table.csv    - Tabela default vs otimizado")

print("\n" + "="*70)
print("✅ PROCESSO FINALIZADO COM SUCESSO!")
print("="*70 + "\n")



# ============================================================================
# PATCH v1.20 → v1.20.1  (Tail-Block Determinatives + Code-Generated Output)
# ----------------------------------------------------------------------------
# Changes applied (per user request):
#  A) Treat '-' as MdC internal sign separator (NOT word boundary). Default: single compound.
#  B) Determinatives selected as contiguous SUFFIX BLOCKS over the whole compound (tail-block extractor).
#  C) LLM is NOT the source of truth for the final set: it only provides JSON evidence; code decides MAIN+ALTS.
#  D) Strong promotion rules for final-location endings (especially O49) + plural clusters immediately before it.
# ============================================================================

# # ---- Config knobs (safe defaults for your evaluation set) ----
# TAIL_WINDOW_SIGNS = 40   # how many *signs* from the end we consider for candidate generation
# SPAN_MAX_LEN = 30        # max determinative-block length (supports long gold outputs like case #4/#5)



# def generate_suffix_span_candidates(words: List[Dict[str, Any]], tokens: List[str], max_len: int = SPAN_MAX_LEN) -> List[Dict[str, Any]]:
#     """
#     v1.20.1 override: suffix spans for the single compound. Uses a tail window to reduce noise,
#     but keeps max_len large enough for the gold outputs.
#     """
#     out: List[Dict[str, Any]] = []
#     if not isinstance(words, list):
#         return out
#     for w in words:
#         if not isinstance(w, dict):
#             continue
#         wid = int(w.get("word_id", 0))
#         idxs = w.get("sign_token_indices") or []
#         idxs = [int(i) for i in idxs if isinstance(i, (int, float)) or (isinstance(i, str) and str(i).isdigit())]
#         idxs = [i for i in idxs if 0 <= i < len(tokens) and tokens[i] not in MDL_SEPARATORS]
#         if not idxs:
#             continue

#         # Tail-window (in *sign indices*), then suffix spans inside that window
#         tail = idxs[-min(TAIL_WINDOW_SIGNS, len(idxs)):]
#         max_k = min(max_len, len(tail))

#         cands: List[Dict[str, Any]] = []
#         cands.append({"span_token_indices": [], "text": "", "length": 0})
#         for k in range(1, max_k + 1):
#             span = tail[-k:]
#             span_tokens = [tokens[i] for i in span]
#             # hard filter NEVER
#             if any(t in NEVER_DETERMINATIVE_SIGNS for t in span_tokens):
#                 continue
#             cands.append({
#                 "span_token_indices": span,
#                 "text": "-".join(span_tokens),
#                 "length": k
#             })
#         out.append({"word_id": wid, "candidates": cands})
#     return out


# def _select_main_and_alts_from_scores(state: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     v1.20.1 override: select MAIN + ALTs from suffix spans of the SINGLE compound.

#     Key promotions:
#       - If final sign is O49: MAIN must include it. Prefer spans that end with ...O49.
#       - If there is a plural run immediately before final O49: reward spans covering the full run.
#       - Penalize contextual-only signs unless supported by evidence.
#       - Prefer shorter spans when scores tie (minimum consistent block).
#     """
#     tokens: List[str] = state.get("tokens") or []
#     mdc = state.get("mdc_structure", {}) or {}

#     # Build words if missing
#     words: List[Dict[str, Any]] = state.get("words") or []
#     if not words:
#         words = build_word_boundaries(tokens, mdc)

#     if not words:
#         return {"main_dets": [], "alt_sets": [], "main_per_word": [], "alt_per_word": []}

#     pool = _gather_candidate_evidence(state)

#     occ_score: Dict[tuple, float] = {}
#     for key, item in pool.items():
#         occ_score[key] = _score_candidate(item)

#     def _as_det_dict(tok: str, idx: int, wid: int) -> Dict[str, Any]:
#         return {
#             "token": tok,
#             "index": int(idx),
#             "gardiner_code": None,
#             "needs_lookup": (tok not in DETERMINATIVES_DETAILED),
#             "word_id": int(wid),
#         }

#     # Determine final sign in the compound
#     sign_stream = [i for i, t in enumerate(tokens) if t not in MDL_SEPARATORS]
#     final_tok = tokens[sign_stream[-1]] if sign_stream else ""
#     final_idx = int(sign_stream[-1]) if sign_stream else None

#     # Compute plural run right before final (only meaningful if final is O49 or O1)
#     def _plural_run_before_end() -> List[int]:
#         if not sign_stream:
#             return []
#         run = []
#         j = len(sign_stream) - 2  # token before final sign
#         while j >= 0:
#             ti = sign_stream[j]
#             if tokens[ti] in PLURAL_MARKERS:
#                 run.append(ti)
#                 j -= 1
#                 continue
#             break
#         return list(reversed(run))

#     plural_run = _plural_run_before_end()

#     def _score_span(span_token_indices: List[int]) -> float:
#         # empty span is valid
#         if not span_token_indices:
#             # If final is a strong location sign, discourage empty
#             if final_tok in FINAL_LOCATION_SIGNS:
#                 return 0.0
#             return 0.02  # tiny bias to allow "Sem determinativos"

#         span_tokens = [tokens[i] for i in span_token_indices if 0 <= i < len(tokens)]
#         s = 0.0

#         # token evidence
#         for idx in span_token_indices:
#             if not (0 <= idx < len(tokens)):
#                 continue
#             tok = tokens[idx]
#             s += occ_score.get(_candidate_key(tok, idx), 0.0)
#             if tok in STRONG_DETERMINATIVE_SIGNS:
#                 s += 0.04

#         # Strong end rules
#         if final_tok == "O49":
#             # MUST include O49 in MAIN-like spans
#             if "O49" not in span_tokens or (span_token_indices and span_token_indices[-1] != final_idx):
#                 s -= 0.8
#             else:
#                 s += 0.25  # strong promotion if it ends with O49

#             # Reward covering plural run before O49 (e.g., Z1Z1Z1Z1-O49)
#             if plural_run:
#                 covered = sum(1 for ti in plural_run if ti in span_token_indices)
#                 if covered == len(plural_run):
#                     s += 0.18
#                 else:
#                     s += 0.06 * (covered / max(1, len(plural_run)))

#         # Location modifier synergy: O31 right before O49/O1 is helpful
#         if "O31" in span_tokens and ("O49" in span_tokens or "O1" in span_tokens):
#             s += 0.08

#         # Penalize contextual-only unless there is solid evidence on that token
#         for tok, idx in zip(span_tokens, span_token_indices):
#             if tok in CONTEXTUAL_ONLY_SIGNS:
#                 base = occ_score.get(_candidate_key(tok, idx), 0.0)
#                 if base < 0.55:  # weak evidence
#                     s -= 0.10

#         # Penalize inclusion of likely phonograms unless final/near-final strong evidence
#         for tok, idx in zip(span_tokens, span_token_indices):
#             if tok in LIKELY_FONOGRAM_SIGNS and tok not in PLURAL_MARKERS:
#                 base = occ_score.get(_candidate_key(tok, idx), 0.0)
#                 if base < 0.65:
#                     s -= 0.12

#         # Length penalty to avoid overprediction (prefer minimal consistent block)
#         s -= 0.015 * max(0, len(span_token_indices) - 1)

#         return max(-2.0, min(3.0, s))

#     # Candidate spans (suffix-only)
#     span_cands = state.get("word_span_candidates")
#     if not isinstance(span_cands, list) or not span_cands:
#         span_cands = generate_suffix_span_candidates(words, tokens, max_len=SPAN_MAX_LEN)

#     # We only expect one word (id=0)
#     wid = int(span_cands[0].get("word_id", 0))
#     candidates = span_cands[0].get("candidates") or []
#     spans = []
#     for c in candidates:
#         if not isinstance(c, dict):
#             continue
#         sp = c.get("span_token_indices") or []
#         if not isinstance(sp, list):
#             continue
#         # validate
#         ok = True
#         for idx in sp:
#             try:
#                 idx = int(idx)
#             except Exception:
#                 ok = False
#                 break
#             if not (0 <= idx < len(tokens)):
#                 ok = False
#                 break
#             if tokens[idx] in MDL_SEPARATORS or tokens[idx] in NEVER_DETERMINATIVE_SIGNS:
#                 ok = False
#                 break
#         if ok:
#             spans.append([int(i) for i in sp])

#     # Score + rank
#     ranked = []
#     for sp in spans:
#         sc = _score_span(sp)
#         dets = [_as_det_dict(tokens[i], i, wid) for i in sp]
#         ranked.append({"score": sc, "span": sp, "det_list": dets, "text": "-".join([tokens[i] for i in sp])})

#     # Dedup by span indices and sort
#     uniq = {}
#     for r in ranked:
#         k = tuple(r["span"])
#         if k not in uniq or r["score"] > uniq[k]["score"]:
#             uniq[k] = r
#     ranked = list(uniq.values())
#     ranked.sort(key=lambda x: (-x["score"], len(x["span"])))

#     if not ranked:
#         return {"main_dets": [], "alt_sets": [], "main_per_word": [], "alt_per_word": []}

#     main = ranked[0]
#     alt_sets = []
#     alt_pw = []
#     main_key = tuple(main["span"])
#     for r in ranked[1:]:
#         if tuple(r["span"]) == main_key:
#             continue
#         alt_sets.append(r["det_list"])
#         alt_pw.append([{"word_id": wid, "span_token_indices": r["span"], "text": r["text"]}])
#         if len(alt_sets) >= 3:
#             break

#     return {
#         "main_dets": main["det_list"],
#         "alt_sets": alt_sets,
#         "main_per_word": [{"word_id": wid, "span_token_indices": main["span"], "text": main["text"]}],
#         "alt_per_word": alt_pw,
#     }

# Note: postprocess_synth_json_with_scoring already uses _select_main_and_alts_from_scores(state).
# With the override above, MAIN/ALTs become deterministic tail-block selections, and the human-readable
# output is generated by format_report_from_synth_json (code), avoiding contradictions.
