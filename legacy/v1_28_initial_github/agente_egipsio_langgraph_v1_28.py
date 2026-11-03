# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 01:31:39 2025

@author: Daniel

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
#import operator
#from typing_extensions import TypedDict

# Configuração de seeds para reprodutibilidade
random.seed(42)
np.random.seed(42)
#torch.manual_seed(42)
#if torch.cuda.is_available():
#    torch.cuda.manual_seed_all(42)

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

DATA_PATH = "E:/estancia_espanha_agentes/content/extractrion_for_agents.xlsx"
OUTPUT_DIR = "E:/estancia_espanha_agentes/content/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    "reliable": {"A1", "B1", "A40", "D54", "Aa2", "Q5", "O49", "O31", "A24", "N1", 
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
    'D54': {'significado': 'pantorrilla y pies', 'categoria': 'action', 'semantica': 'ir, caminar, viajar, entrar, carrera, procesión'},
    'D3': {'significado': 'mechones de pelo', 'categoria': 'body', 'semantica': 'pelo, cabello, peluquero, peluca, viuda, luto, lamento, calvo'},
    'Aa2': {'significado': 'pústula o glándula (?)', 'categoria': 'abstract', 'semantica': 'embalsamador, descomponerse, herida, dañino, malo'},
    
    # ANIMALES
    'E1': {'significado': 'ganado vacuno', 'categoria': 'animal', 'semantica': 'ganado vacuno, toro, vaca, ganado mayor'},
    'E8': {'significado': 'cabrito', 'categoria': 'animal', 'semantica': 'cabrito, ganado menor'},
    'F27': {'significado': 'piel', 'categoria': 'body', 'semantica': 'piel, cuero, piel animal, correa, taparrabos, ratón, gato'},
    'G7': {'significado': 'halcón sobre estandarte', 'categoria': 'divine', 'semantica': 'dios, rey'},
    'G39': {'significado': 'pato', 'categoria': 'animal', 'semantica': 'pato, ave, garza, langosta'},
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
    'O1': {'significado': 'plano de casa', 'categoria': 'building', 'semantica': 'casa, edificio, localidad'},
    
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
    'T18': {'significado': 'cayado', 'categoria': 'object', 'semantica': 'seguir, partidario, acompañante'},
    
    # V1.6: SIGNOS ADICIONADOS (baseado em análise de falsos negativos)
    'P5B': {'significado': 'barco (variante)', 'categoria': 'transport', 'semantica': 'navegación, viaje por agua, barca'},  # Perdido 2x
    'V1': {'significado': 'lazo', 'categoria': 'object', 'semantica': 'lazo, unir, atar, cien mil'},  # Perdido 2x
    'A193': {'significado': 'persona (variante)', 'categoria': 'person', 'semantica': 'persona, agente, variante de A1'},  # (duplicado removido, já adicionado acima)
    'O36': {'significado': 'muro alto con abertura', 'categoria': 'building', 'semantica': 'fortaleza, castillo, muro'},  # Comum mas pode não estar
    'A7A': {'significado': 'persona sentada (variante)', 'categoria': 'person', 'semantica': 'persona, variante'},  # Mencionado em casos
    'A24': {'significado': 'hombre con brazo extendido', 'categoria': 'person', 'semantica': 'acción con el brazo, golpear'},  # Perdido 1x
    'A35': {'significado': 'hombre construyendo', 'categoria': 'person', 'semantica': 'constructor, trabajador, construir'},  # Perdido 1x
    'D20': {'significado': 'nariz', 'categoria': 'body', 'semantica': 'nariz, oler, respirar, alegría'},  # Perdido 1x
    'W24': {'significado': 'taza', 'categoria': 'object', 'semantica': 'vaso, recipiente, ofrecer'},  # Perdido 1x
    'X4': {'significado': 'pan', 'categoria': 'food', 'semantica': 'pan, ofrenda'},  # Perdido 1x
    'Y2': {'significado': 'papiro enrollado', 'categoria': 'abstract', 'semantica': 'libro, documento, palabra escrita'},  # Perdido 1x
    'Q5': {'significado': 'sarcófago', 'categoria': 'funeral', 'semantica': 'sarcófago, ataúd, muerte'},
    
    # ========================================================================
    # V1.27: EXPANSÃO SISTEMÁTICA DA BASE DE CONHECIMENTO (+70 sinais)
    # Baseado em análise de FN e categorias completas de Gardiner
    # ========================================================================
    
    # SÉRIE F: PARTES DO CORPO (faltantes)
    'F51': {'significado': 'pedaço de carne com sangue', 'categoria': 'body', 
            'semantica': 'carne, sangue, açougueiro, sacrifício, corpo'},
    
    # SÉRIE N: PAISAGEM/ÁGUA (faltantes)
    'N16': {'significado': 'terreno cultivado', 'categoria': 'land', 
            'semantica': 'terra, campo, parcela, agricultura'},
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
                model="gpt-5-mini",
                temperature=0.0,
                top_p=1,
                seed=1234,
                api_key=os.environ.get('OPENAI_API_KEY')
            )
        elif 'ANTHROPIC_API_KEY' in os.environ:
            llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.4,
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

SEQUENCE (MdC):
{state['mdc_sequence']}

TOKENS WITH INDICES (0-based, includes separators):
{list(enumerate(state['tokens']))}

MdC STRUCTURE HINTS (computed) + WORD UNITS:
PHONETIC COMPLEMENT HINTS (computed; treat as complements by default):
{state.get('phonetic_complement_hints', [])}

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
                model="gpt-5-mini",
                temperature=0.0,
                top_p=1,
                seed=1234,
                api_key=os.environ.get('OPENAI_API_KEY')
            )
        elif 'ANTHROPIC_API_KEY' in os.environ:
            llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.3,
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

        human_prompt = f"""SEQUENCE (MdC):
{state['mdc_sequence']}

TOKENS WITH INDICES (0-based, includes separators):
{list(enumerate(state['tokens']))}

MdC STRUCTURE HINTS (computed) + WORD UNITS:
PHONETIC COMPLEMENT HINTS (computed; treat as complements by default):
{state.get('phonetic_complement_hints', [])}

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
- Identify determinatives supported by structural/positional patterns.
- Set rule_agree=true when baseline also supports the same token+index.
- Return JSON ONLY, strictly matching the schema.
"""

        # Task do agente
        human_prompt = f"""{state['expert_context']}
STRUCTURAL ANALYSIS — evaluate multiple criteria

SEQUENCE: {state['mdc_sequence']}
TOKENS WITH POSITIONS (0-based): {list(enumerate(state['tokens']))}

CANDIDATES:
{state['candidates_text']}

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
                model="gpt-5-mini",
                temperature=0.0,
                top_p=1,
                seed=1234,
                api_key=os.environ.get('OPENAI_API_KEY')
            )
        elif 'ANTHROPIC_API_KEY' in os.environ:
            llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.2,
                api_key=os.environ.get('ANTHROPIC_API_KEY')
            )
        
        if llm is None:
            state['consolidated_analysis'] = ""
            state['determinatives'] = []
            return state
        # PROMPT (SYNTHESIZER) — structured Spanish report in JSON (v3.1)
        system_prompt = """You are a senior Egyptologist acting as the FINAL ARBITER.

GOAL:
Integrate 3 sources (Semantic JSON, Structural JSON, Rule-based baseline) and produce (PER WORD/COMPOUND when possible):
- a MAIN interpretation with an explicit determinative set (possibly empty),
- a set of ALTERNATIVE interpretations (if plausible),
- an AMBIGUITY summary,
- a RECOMMENDATION.

MANDATORY RULES:
1) Use ONLY tokens present in the provided token list; NEVER invent signs.
1b) Use provided WORD/COMPOUND units to decide 'final of word' positions; determinatives are typically final of a word/compound.
1c) Determinatives MUST be selected as CONTIGUOUS SUFFIX SPANS per word/compound (a 'determinative block'). Do NOT pick isolated internal tokens; if evidence is weak, return an empty span for that word.
1d) Z1/Z2/Z3 may appear inside a determinative block when adjacent to the suffix; keep repetitions with their indices.
2) Never output separators as determinatives: '-', ':', '*'.
3) Repetitions matter: include EACH occurrence separately with its token index (0-based).
4) Prioritize precision; do NOT force determinatives when evidence is weak.
6) IMPORTANT: Determinatives may appear in NON-FINAL positions in formulaic or enumerative sequences; do not over-weight "final position" when semantic+rules strongly support non-final candidates.
5) Output MUST be valid JSON only (no markdown, no extra text).


EGYPTOLOGICAL PRINCIPLES (Middle Egyptian):
A) Mixed system: phonograms/logograms/determinatives. Output determinatives/classifiers only.
B) Phonetic complements (CRITICAL): common uniliterals near the end of a word are often phonetic complements, not determinatives. Do NOT promote them to determinatives unless strongly supported.
C) Layering: core → complements → determinatives/classifiers → plural strokes. Treat Z1/Z2/Z3 as functional markers when present.
D) Alternatives must be actionable: include a minimal PROMOTION CONDITION inside each alternative's 'reason' (what extra context would make it MAIN).


OUTPUT LANGUAGE:
Write all narrative fields (justification, validation, reasons, ambiguity, recommendation) in SPANISH.
Keep sign codes (e.g., M17, R16) unchanged.

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

        human_prompt = f"""SYNTHESIS — integrate sources and decide final output

SEQUENCE (MdC):
{state['mdc_sequence']}

TOKENS WITH INDICES (0-based, includes separators):
{list(enumerate(state['tokens']))}

MdC STRUCTURE HINTS (computed) + WORD UNITS:
PHONETIC COMPLEMENT HINTS (computed; treat as complements by default):
{state.get('phonetic_complement_hints', [])}

PHONETIC COMPLEMENT HINTS (computed; treat as complements by default):
{state.get('phonetic_complement_hints', [])}

- block_final_token_indices: {state.get('mdc_structure', {}).get('block_final_token_indices', [])}
- blocks: {state.get('mdc_structure', {}).get('blocks', [])}
- groups: {state.get('mdc_structure', {}).get('groups', [])}
- words (probable word/compound units): {state.get('words', [])}
- suffix span candidates per word (determinative blocks): {state.get('word_span_candidates', [])}

SEMANTIC JSON (parsed):
{json.dumps(state.get('semantic_json', {}), ensure_ascii=False)}

STRUCTURAL JSON (parsed):
{json.dumps(state.get('structural_json', {}), ensure_ascii=False)}

RULE-BASED BASELINE (validated):
{state['rule_result']}

TASK:
1) MAIN: Provide an explicit determinative set (possibly empty) and a clear Spanish justification.
2) ALTERNATIVES: If there are plausible alternative sets, include them; otherwise an empty list.
3) AMBIGUITY: Summarize key ambiguity factors and consensus level in Spanish.
4) RECOMMENDATION: Give practical bullets in Spanish (no need to mention confidence/grades).

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
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
        else:
            llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0, top_p=1, seed=1234)
        
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
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1)
        else:
            llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0, top_p=1,seed=1234)
        
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

def analyze_with_langgraph(mdc_sequence: str, image_file: str = ""):
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
        
        # Contexto de exemplos do especialista
        expert_context = format_expert_examples(EXPERT_EXAMPLES) if EXPERT_EXAMPLES else ""
        
        # Criar enriched_determinatives ANTES dos agentes (sempre executado)
        # CORREÇÃO v1.26: Linha comentada - causa FP ao adicionar símbolos desconhecidos via LLM
        # O gardiner_lookup_agent não tem contexto suficiente da sequência e detecta símbolos incorretamente
        # enriched_determinatives = enrich_determinatives_with_lookup(tokens, DETERMINATIVES_DETAILED)
        enriched_determinatives = DETERMINATIVES_DETAILED  # Usar apenas base de conhecimento fixa
        
        # Preparar estado inicial
        initial_state = {
            'mdc_sequence': mdc_sequence,
            'image_file': image_file,
            'tokens': tokens,
            'candidates_info': candidates_info,
            'candidates_text': candidates_text,
            'rule_result': rule_result,
            'expert_context': expert_context,
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

# Processar quantidade/range configurado de frases
if INICIO_RANGE is not None or FIM_RANGE is not None:
    # MODO RANGE: processar faixa específica
    inicio = INICIO_RANGE if INICIO_RANGE is not None else 0
    fim = FIM_RANGE if FIM_RANGE is not None else len(df)
    
    df_to_process = df.iloc[inicio:fim]
    
    print(f"📊 Processando RANGE de frases: {inicio} a {fim}")
    print(f"   Total: {len(df_to_process)} frases (de {len(df)} totais)")
    
elif NUM_FRASES_PROCESSAR is None:
    # MODO TODAS: processar todo o dataset
    df_to_process = df
    print(f"📊 Processando TODAS as {len(df)} frases do dataset")
    
else:
    # MODO QUANTIDADE: processar N primeiras frases
    df_to_process = df.head(NUM_FRASES_PROCESSAR)
    print(f"📊 Processando {len(df_to_process)} primeiras frases (de {len(df)} totais)")

# Processar cada sequência
#for idx, row in df.iterrows():
for idx, row in df_to_process.iterrows():
    mdc_sequence = row['Manuel_de_Codage']
    image_file = row.get('image_file', f'img_{idx}')
    
    if pd.isna(mdc_sequence) or mdc_sequence == '':
        continue
    
    if USE_LANGGRAPH:
        # Análise com LangGraph
        result = analyze_with_langgraph(mdc_sequence, image_file)
        
        if result['success']:
            df.at[idx, 'determinative_count'] = len(result['determinatives'])
            df.at[idx, 'determinatives_found'] = result['determinatives']
            df.at[idx, 'analysis_method'] = 'multi_agent'
            df.at[idx, 'langgraph_analysis'] = result['consolidated_analysis']
        else:
            # Fallback para regras
            tokens = tokenize_mdc(mdc_sequence)
            rule_result = analyze_rule_based(tokens)
            df.at[idx, 'determinative_count'] = len(rule_result['determinatives'])
            df.at[idx, 'determinatives_found'] = rule_result['determinatives']
            df.at[idx, 'analysis_method'] = 'rule_based'
    else:
        # Análise apenas por regras
        tokens = tokenize_mdc(mdc_sequence)
        rule_result = analyze_rule_based(tokens)
        df.at[idx, 'determinative_count'] = len(rule_result['determinatives'])
        df.at[idx, 'determinatives_found'] = rule_result['determinatives']
        df.at[idx, 'analysis_method'] = 'rule_based'
    
    # Mostrar progresso
    if (idx + 1) % 50 == 0:
        print(f"  Processadas: {idx + 1}/{len(df)}")

print("\n✅ Processamento concluído!")
print(f"   Sequências processadas: {len(df)}")
print(f"   Com LangGraph: {(df['analysis_method'] == 'multi_agent').sum()}")
print(f"   Apenas regras: {(df['analysis_method'] == 'rule_based').sum()}")

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
    for i, (sign, count) in enumerate(top_10, 1):
        info = DETERMINATIVES_DETAILED.get(sign, {})
        significado = info.get('significado', 'desconocido')
        print(f"   {i}. {sign}: {count}x - {significado}")
    
    # Gráfico de barras
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
    for cat, count in category_counts.most_common():
        pct = (count / total_det) * 100
        print(f"   {cat}: {count} ({pct:.1f}%)")
    
    # Gráfico de pizza
    plt.figure(figsize=(10, 8))
    categories, counts = zip(*category_counts.most_common())
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
for idx, row in df.iterrows():
    if row['determinative_count'] > 0:
        for det in row['determinatives_found']:
            # Buscar informações detalhadas do determinativo
            sign_code = det['sign']
            det_info = DETERMINATIVES_DETAILED.get(sign_code, {})
            
            export_data.append({
                'index': idx,
                'image_file': row.get('image_file', ''),
                'mdc_sequence': row['Manuel_de_Codage'],
                'determinative_sign': sign_code,
                'position': det['position'],
                'significado': det_info.get('significado', 'desconocido'),     # ✅
                'categoria': det_info.get('categoria', 'Desconocido'),         # ✅
                'semantica': det_info.get('semantica', 'N/A'),                 # ✅
                'confidence': det.get('confidence', 'unknown'),
                'method': det.get('method', 'unknown'),
                'expert_translation': row.get('expert_translation', '')
            })

df_export = pd.DataFrame(export_data)
csv_path = f"{OUTPUT_DIR}/determinatives_detailed.csv"
df_export.to_csv(csv_path, index=False, encoding='utf-8')
print(f"✅ CSV exportado: {csv_path}")

# 2. Dataset completo com colunas de determinativos
full_export_path = f"{OUTPUT_DIR}/dataset_with_determinatives.csv"
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
print(f"   • Sequências processadas: {len(df)}")
print(f"   • Determinativos: {df['determinative_count'].sum()}")
print(f"   • Método usado: {'LangGraph Multi-Agente' if USE_LANGGRAPH else 'Apenas Regras'}")
print(f"   • Arquivos gerados: {OUTPUT_DIR}")

if USE_LANGGRAPH:
    print("\n🤖 SISTEMA LANGGRAPH:")
    print(f"   • Sequências analisadas: {(df['analysis_method'] == 'multi_agent').sum()}")
    print(f"   • Base de conhecimento: {len(DETERMINATIVES_DETAILED)} determinativos")
    print("   • Agentes: Semanticista, Estruturalista, Sintetizador")

print("\n📁 ARQUIVOS DISPONÍVEIS:")
print("   • determinatives_detailed.csv - Tabela detalhada")
print("   • dataset_with_determinatives.csv - Dataset completo")
print("   • top10_determinatives.png - Gráfico de frequência")
print("   • categories_distribution.png - Distribuição por categoria") 
if USE_LANGGRAPH:
    print("   • langgraph_analyses.json - Análises do sistema (JSON)")
    print("   • langgraph_analyses.txt - Análises do sistema (legível)")

print("\n" + "="*70)
print("✅ PROCESSO FINALIZADO COM SUCESSO!")
print("="*70 + "\n")
