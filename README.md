# Proyecto 1: Monitoreo Transaccional — Detección de Fraude con Deep Learning

**Integrantes:** Jonathan Zacarías (Carné 231104) | Luis Gilberto (Carné 23353)  
**Curso:** Deep Learning & Redes Recurrentes  
**Pregunta Central de Investigación:**  
> *«¿El orden en que ocurren las transacciones aporta información que las variables agregadas (promedios, conteos por hora, etc.) no capturan — y cuánto vale eso en quetzales?»*

---

## 1. Resumen Ejecutivo y Resultados Clave

| Modelo / Sistema | Enfoque Arquitectónico | Umbral Óptimo ($\theta^*$) | AUC-PR | AUC-ROC | F1-Score | Recall | Precisión | Costo Test (15 días) | Ahorro Mensual Proyectado |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sistema Ingenuo** | Tabular Baseline sin calibrar | 0.50 | 0.542 | 0.812 | 0.447 | 32.1% | 73.5% | Q114,840.00 | *Referencia base* |
| **Modelo A (Agregado)** | MLP Tabular (12 variables) | 0.22 | 0.542 | 0.812 | 0.528 | 65.4% | 44.2% | Q91,140.00 | +Q47,400.00 / mes |
| **Modelo B (Secuencial)** | LSTM Recurrente + Embeddings | 0.20 | **0.836** | **0.958** | **0.768** | **83.2%** | **71.3%** | **Q67,680.00** | **+Q46,920.00 / mes** |
| **Apuesta C (Híbrido)** | Dual-Branch Net (Secuencia + MLP) | 0.21 | **0.841** | **0.961** | **0.773** | **84.5%** | **71.2%** | **Q67,020.00** | **+Q48,240.00 / mes** |

### Conclusiones Centrales:
1. **El valor del orden es causal e irrefutable:** La prueba de falsificación por permutación intra-secuencia provocó un **colapso del 36.6% en AUC-PR** en el Modelo B, demostrando que el modelo aprovecha genuinamente la cronología y no atajos estadísticos.
2. **Impacto Financiero:** El despliegue de la arquitectura secuencial/híbrida genera un **ahorro de más de Q48,000 mensuales** para la institución bancaria bajo la función de pérdida asimétrica ($FN = \text{Q4,200}$, $FP = \text{Q180}$).

---

## 2. Estructura Real del Repositorio

El proyecto está diseñado como un **entregable autocontenido**, donde todo el pipeline (generación de datos sintéticos, ingeniería de variables, arquitecturas PyTorch, entrenamiento, pruebas de falsificación y análisis económico) se encuentra implementado y ejecutable dentro del Jupyter Notebook principal:

```text
ProyectoDeepLearning/
├── proyecto1_monitoreo_transaccional.ipynb     # Notebook PRINCIPAL (100% Autocontenido)
├── informe.pdf                                # Informe técnico y de negocio formal (6 evidencias)
├── presentacion.pdf                           # Diapositivas ejecutivas del proyecto
├── README.md                                  # Documentación del proyecto, decisiones y API contract
└── artefactos/                                 # Artefactos exportados durante la ejecución
    ├── modelo_a_baseline.pt                   # Pesos serializados del Modelo A (MLP Tabular)
    ├── modelo_b_lstm.pt                       # Pesos serializados del Modelo B (LSTM Secuencial)
    ├── modelo_c_hybrid.pt                     # Pesos serializados de la Apuesta C (Red Híbrida)
    ├── curvas_pr_roc.png                      # Curvas Precision-Recall y ROC en Test Set
    ├── prueba_permutacion.png                 # Gráfica de la Prueba de Permutación Controlada
    ├── desglose_mecanismos_fraude.png         # Eficacia de detección por mecanismo de fraude
    ├── curva_costo_economico.png              # Curvas de costo total en Quetzales
    └── metricas_completas.json                # Resumen numérico completo en JSON
```

---

## 3. Instrucciones de Reproducción

### Requisitos del Entorno
- **Python:** `>= 3.10` (probado en Python 3.10, 3.11 y 3.12)
- **PyTorch:** `torch >= 2.0.0`
- **Scikit-Learn:** `scikit-learn >= 1.3.0`
- **Pandas:** `pandas >= 2.0.0`
- **NumPy:** `numpy >= 1.24.0`
- **Visualización:** `matplotlib >= 3.7.0`, `seaborn >= 0.12.0`
- **Entorno:** Jupyter Notebook, JupyterLab o Google Colab

### Ejecución
1. Abre el notebook principal [proyecto1_monitoreo_transaccional.ipynb](file:///c:/Users/jonat/Desktop/ProyectoDeepLearning/proyecto1_monitoreo_transaccional.ipynb) en tu entorno preferido (VS Code, Jupyter o Google Colab).
2. Ejecuta todo el flujo haciendo clic en **"Restart Kernel and Run All"** (Reiniciar y Ejecutar Todo).
3. Todas las dependencias de datos, clases, modelos y funciones se construyen secuencialmente en las celdas sin requerir librerías locales ni archivos externos.

---

## 4. Declaración de Uso de Inteligencia Artificial (IA)

En concordancia con los principios de integridad académica y transparencia técnica:

- **Para qué se utilizó IA:**
  - Asistencia en la estructuración de visualizaciones con `matplotlib`/`seaborn` y formateo de tablas en Markdown.
  - Planteamiento preliminar de la formulación matricial de costos financieros asimétricos en Quetzales.
- **Qué verificamos nosotros mismos:**
  - **Protocolo temporal estricto:** Verificación manual de que no existe fuga de datos (`Train` $\rightarrow$ `Val` $\rightarrow$ `Test`) y que los escaladores/encoders se ajustan exclusivamente con datos de `Train`.
  - **Diseño de pruebas de falsificación:** Implementación de la permutación aleatoria intra-secuencia y validación empírica del colapso en AUC-PR.
  - **Alineación de hiperparámetros y convergencia:** Ponderación de pérdida `pos_weight` ante desbalance severo de clases (~3.5%), control de sobreajuste con dropout y regularización L2.
  - **Revisión analítica y financiera:** Interpretación del costo financiero real, formulación del contrato de API y redacción del informe técnico.

---

## 5. Tres Decisiones Técnicas Importantes

### Decisión 1: Adopción de la Ruta A (Generación Sintética Controlada) con Inyección Causal
- **Alternativas consideradas:** Datasets públicos (IEEE-CIS / PaySim / Kaggle Credit Card) vs. Entorno sintético controlado.
- **Evidencia que inclinó la decisión:** Para aislar formalmente si el orden temporal aporta valor real, era indispensable controlar qué fraudes dependían críticamente de la secuencia (`probing_drainage`, `velocity_shift`) y cuáles eran independientes del orden (`amount_spike`). Los datasets públicos anonimizados sufren de marcas de tiempo alteradas o falta de historial granular por cliente. Con la Ruta A se logró un control causal perfecto sin variables confensoras.

### Decisión 2: Optimización del Umbral de Decisión $\theta^*$ Exclusivamente en Validación
- **Alternativas consideradas:** Umbral por defecto de 0.50 vs. Optimización en Test vs. Optimización en Validación con función de costo asimétrica.
- **Evidencia que inclinó la decisión:** Un falso negativo cuesta 23.3 veces más que un falso positivo (Q4,200 vs Q180). Calibrar el umbral en Test violaría el principio de evaluación ciega. Por tanto, se construyó la curva de costo en Validación para hallar el umbral óptimo ($\theta^* \approx 0.20$), aplicándolo ciegamente a Test para maximizar el ahorro económico real sin sesgo retrospectivo.

### Decisión 3: Arquitectura Híbrida Dual-Branch (Apuesta C) en lugar de un Enfoque Secuencial Aislado
- **Alternativas consideradas:** Modelo LSTM puro vs. Detección no supervisada (Autoencoder) vs. Red Híbrida Dual-Branch.
- **Evidencia que inclinó la decisión:** El fraude bancario es un fenómeno multi-escala: la rama LSTM con Embeddings captura la dinámica de la micro-secuencia inmediata ($K=6$), mientras que la rama MLP aporta el contexto macro-histórico del cliente a 60 días (`hist_avg_amount`, `amount_to_hist_avg_ratio`). La fusión de ambas ramas superó el umbral de suficiencia declarado ($\Delta\text{AUC-PR} \ge +0.05$ y ahorro $\ge \text{Q15,000/mes}$) logrando **+0.299 en AUC-PR** y **~Q48,200/mes de ahorro**.

---

## 6. Candidato al Proyecto Final y Contrato de API

### 1. Modelo Seleccionado y Ubicación del Artefacto
- **Modelo Principal:** `Apuesta C: HybridDualBranchNet` (y `Modelo B: SequentialFraudLSTM` como alternativa ligera).
- **Ubicación del Artefacto:** `artefactos/modelo_c_hybrid.pt` y `artefactos/metricas_completas.json`.

### 2. Usuario del Puntaje y Decisión Operativa
- **Usuario Operativo:** Motor de Autorización Transaccional en Línea y Analistas de Prevención de Fraude.
- **Políticas de Decisión:**
  - $P(\text{Fraude}) < 0.18$: **Aprobación inmediata** (transacción normal sin fricción).
  - $0.18 \le P < 0.65$: **Autenticación Reforzada (Step-Up / 2FA)** mediante OTP por SMS o biometría facial en app bancaria.
  - $P \ge 0.65$: **Bloqueo preventivo de la tarjeta** y alerta prioritaria al analista de fraude.

### 3. Contrato Preliminar de Entrada y Salida (API Schema)

```json
// ENTRADA (Payload JSON recibido por el endpoint de inferencia)
{
  "client_id": "CLI_0142",
  "current_transaction": {
    "tx_id": "TX_0089123",
    "timestamp": "2026-03-31T14:23:10Z",
    "amount": 4500.00,
    "merchant_category": "LUXURY_JEWELRY",
    "channel": "ONLINE",
    "is_international": 1
  },
  "recent_history": [
    {"timestamp": "2026-03-31T14:15:02Z", "amount": 10.00, "merchant_category": "DIGITAL_SERVICE", "channel": "ONLINE", "is_international": 1},
    {"timestamp": "2026-03-31T14:18:22Z", "amount": 12.50, "merchant_category": "DIGITAL_SERVICE", "channel": "ONLINE", "is_international": 1},
    {"timestamp": "2026-03-31T14:20:45Z", "amount": 15.00, "merchant_category": "DIGITAL_SERVICE", "channel": "ONLINE", "is_international": 1}
  ],
  "aggregated_profile_24h_7d": {
    "hist_avg_amount": 210.50,
    "roll_mean_5": 12.50,
    "roll_max_5": 15.00,
    "roll_std_5": 2.50,
    "delta_time_hours": 0.040,
    "roll_intl_count_5": 3
  }
}

// SALIDA (Respuesta JSON emitida en <25 ms)
{
  "tx_id": "TX_0089123",
  "fraud_risk_score": 0.9421,
  "decision": "BLOCK_TRANSACTION",
  "recommended_action": "TRIGGER_IMMEDIATE_CARD_LOCK_AND_ALERT",
  "detected_pattern": "probing_drainage_sequence",
  "confidence": 0.96
}
```

### 4. Límites, Riesgos y Datos Faltantes para Producción
- **Límites:** Sensibilidad ante clientes con viajes al extranjero sin notificación previa (*falsos positivos* mitigados mediante autenticación 2FA).
- **Riesgos:** Latencia de red si el servicio de búsqueda de historia reciente no cuenta con almacenamiento en memoria (*In-Memory Cache* como Redis).
- **Datos faltantes para Fase de Producción:**
  - Huella digital de dispositivo (`device_id`, IP, geolocalización GPS, cambio de SIM).
  - Estado del saldo de cuenta en tiempo real.
  - Historial de contracargos previos del tarjetahabiente.
