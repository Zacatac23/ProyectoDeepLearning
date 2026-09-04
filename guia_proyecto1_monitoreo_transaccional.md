# Guía de trabajo — Proyecto 1: Monitoreo Transaccional
### Cómo abordarlo por fases sin perderse

---

## Idea central que no debes perder de vista

Todo el proyecto responde a **una sola pregunta**:

> ¿El orden en que ocurren las transacciones aporta información que las variables agregadas (promedios, conteos por hora, etc.) no capturan — y cuánto vale eso en quetzales?

Cada fase que sigue existe para responder un pedazo de esa pregunta. Si en algún momento no sabes por qué estás haciendo algo, vuelve a esta pregunta.

---

## Fase 0 — Antes de escribir código

**Objetivo:** decidir el terreno de juego antes de jugar.

- [ ] Formar la pareja y repartir roles (aunque ambos deben poder defender cualquier parte).
- [ ] Elegir **Ruta A (datos sintéticos)** o **Ruta B (datos públicos reales)**. Ver criterios abajo.
- [ ] Crear la estructura de carpetas del repo:
  ```
  proyecto1_<apellidos>/
  ├── proyecto1_<apellidos>.ipynb
  ├── artefactos/
  ├── informe.pdf
  ├── presentacion.pdf
  └── README.md
  ```
- [ ] Leer la rúbrica y los descuentos (sección 6 del enunciado) **antes** de programar. Son errores que se previenen con diseño, no se corrigen después.

**Cómo elegir ruta:**

| Si prefieres... | Elige |
|---|---|
| Controlar exactamente qué patrón de fraude depende del orden, y poder demostrarlo con certeza | **Ruta A** (sintética) |
| Trabajar con ruido y ambigüedad real, sin poder "hacer trampa" sabiendo la respuesta de antemano | **Ruta B** (pública) |

Ambas valen lo mismo. Ruta A es más controlable pedagógicamente; Ruta B es más creíble ante el "comité de riesgos" pero más difícil de interpretar.

---

## Fase 1 — Los datos (la base de todo)

**Objetivo:** tener secuencias de transacciones por cliente/tarjeta, ordenadas en el tiempo, con etiqueta de fraude.

### Si Ruta A (sintética):
- Define **al menos 3 tipos de fraude**. Al menos uno debe depender del orden (ej: varias compras pequeñas → una grande, o cambio súbito de comercio/canal).
- El generador debe ser **reproducible con semilla fija** (`np.random.seed(...)` o similar).
- Escribe en el notebook, en palabras simples, qué patrón representa cada tipo de fraude.
- Incluye deliberadamente **al menos un caso donde esperas que el modelo falle** — esto no es opcional, es evidencia de honestidad analítica.

### Si Ruta B (pública real):
- Busca un dataset con: identificador de cliente/tarjeta, timestamp, y etiqueta de fraude (ej. datasets de Kaggle de fraude con tarjetas, IEEE-CIS, PaySim, etc.).
- Verifica que puedas **agrupar transacciones por entidad** para formar secuencias.
- Documenta: cuántas transacciones tiene cada secuencia en promedio, qué representa una secuencia, qué es exactamente lo que el modelo predice (¿la siguiente transacción es fraude? ¿la secuencia contiene fraude?).

### En ambas rutas — regla de oro temporal:
```
[ ENTRENAR: más antiguo ] → [ VALIDAR: intermedio ] → [ PROBAR: más reciente ]
```
- **Nunca** mezcles el futuro con el pasado (esto es fuga de información y cuesta -20 puntos si es aleatoria).
- El conjunto de prueba se toca **una sola vez**, al final, después de fijar TODAS las decisiones (arquitectura, umbral, hiperparámetros).
- Cualquier normalización o selección de variables se calcula **solo con el set de entrenamiento**, nunca con el dataset completo (-15 puntos si no).

---

## Fase 2 — Modelo A: la línea base sin orden

**Objetivo:** establecer "el piso" — qué tan bien se puede detectar fraude *sin leer la secuencia*, solo con variables agregadas (monto promedio 24h, transacciones por hora, monto máximo del día, diversidad de comercios, etc.).

- Elige un modelo simple y defendible: regresión logística, árboles/XGBoost, red densa (MLP).
- La complejidad no da puntos aquí — la claridad sí.
- Debe devolver un **puntaje continuo de riesgo** (no una clase binaria todavía).
- Este modelo es tu punto de comparación para todo lo que sigue.

**Pregunta que debes poder responder al terminar esta fase:** *"¿Qué tan bien funciona el sistema actual del banco, en esencia?"*

---

## Fase 3 — Modelo B: el modelo secuencial

**Objetivo:** ver si "leer el orden" mejora la detección.

- Arquitectura sugerida: LSTM, GRU, RNN simple, CNN temporal/TCN, o Transformer (atención). Elige una y justifica por qué, no uses la más compleja "porque sí".
- Debe recibir los eventos **en su orden real**, no agregados.
- Mismo output: puntaje continuo de riesgo.
- Usa **los mismos datos, misma partición, mismo horizonte de predicción** que A. Esto es lo que hace la comparación válida.

**Pregunta que debes poder responder:** *"¿B supera a A? ¿Por cuánto, y en qué métrica?"* (AUC-PR, no exactitud — el problema está desbalanceado).

---

## Fase 4 — Las dos pruebas de falsificación (la parte más importante)

Este es el corazón intelectual del proyecto. **Una mejora de métricas NO prueba que el modelo usó el orden.** Hay que intentar activamente romper tu propia conclusión.

### Prueba obligatoria: Permutación controlada
1. Toma el modelo B ya entrenado.
2. Baraja el orden de los eventos **dentro de cada secuencia** (sin tocar las variables agregadas ni los eventos mismos).
3. Evalúa B sobre estas secuencias barajadas.
4. Compara contra el desempeño con el orden original.

**Interpretación honesta:**
- Si el desempeño **cae** al barajar → el modelo sí estaba usando el orden. ✅
- Si el desempeño **no cae** → no puedes afirmar que el orden aportó, aunque B haya sido mejor que A por otras razones (ej. B capturó interacciones no lineales que A no podía, sin necesidad de orden).

### Prueba elegida por el equipo (escoge una)
- Recortar la historia (secuencias más cortas).
- Retirar las variables temporales.
- Evaluar por mecanismo de fraude (¿B es mejor solo en el tipo de fraude que depende del orden?).
- Simular cambio de canal/comportamiento.
- Probar con secuencias más largas que las de entrenamiento.
- Otra prueba que pueda **hacer fallar tu afirmación** — ese es el criterio: debe ser una prueba que honestamente podría refutarte, no una que confirme lo que ya crees.

---

## Fase 5 — La apuesta C (tu hipótesis propia)

**Objetivo:** demostrar pensamiento experimental real, no solo ejecución del núcleo obligatorio.

**Antes de entrenar nada de C**, escribe en el notebook esta frase completa:

> "Creemos que ___ mejorará ___ porque ___. Lo consideraremos útil si ___."

Ejemplos de dirección (no obligatorios, son ideas):
- Modelo híbrido que combine agregados (A) + secuencia (B).
- Detección no supervisada de fraudes nuevos (anomalías no vistas en entrenamiento).
- Embeddings de comercio/canal.
- Atención interpretable (visualizar qué transacciones "pesan más").
- Aprendizaje con pocas etiquetas.
- Robustez ante un mecanismo de fraude no visto en entrenamiento.

**Reglas:**
- Necesita un **control experimental** (comparar contra algo, no solo mostrar un número suelto).
- Necesita una **métrica de éxito declarada antes de ver el resultado**.
- Si falla, **igual se reporta y se califica** — el veredicto honesto vale más que un éxito forzado.

---

## Fase 6 — Decisión económica (traducir el modelo a quetzales)

**Objetivo:** convertir métricas en una decisión de negocio.

Datos dados:
- Fraude no detectado (falso negativo) cuesta en promedio **Q4,200**.
- Bloquear una transacción legítima (falso positivo) cuesta **Q180**.

Pasos:
1. Con el puntaje continuo de riesgo (de A y/o B), construye la curva de costo total en función del umbral de decisión.
   ```
   Costo total(umbral) = (# falsos negativos × Q4,200) + (# falsos positivos × Q180)
   ```
2. Encuentra el umbral que minimiza el costo total en el set de **validación** (no en el de prueba).
3. Aplica ese umbral fijo al set de prueba y reporta el ahorro/pérdida mensual estimado comparado con el sistema actual (o con un umbral ingenuo).

Esto es lo que hace que el informe hable "el idioma" del comité de riesgos, no el del profesor.

---

## Fase 7 — El informe (máx. 7 páginas, sin código)

Debe contener, localizables sin adivinar, estas **seis evidencias**:

1. **Integridad de datos** — origen, tamaño, tasa de fraude, cómo se construyeron las secuencias, partición temporal, controles anti-fuga.
2. **Comparación común A vs B** — AUC-PR + precisión/exhaustividad/F1 en el umbral elegido. (Nunca exactitud como métrica principal).
3. **Valor del orden** — resultados de las dos pruebas de falsificación, interpretados con honestidad.
4. **Apuesta del equipo** — hipótesis previa, control, resultado, veredicto (aunque haya fallado).
5. **Decisión económica** — umbral elegido y ahorro/pérdida mensual estimado.
6. **Recomendación y límites** — ¿reemplazar, complementar o conservar el sistema actual? Al menos un patrón de error concreto + condiciones bajo las que cambiarías de opinión.

Cierra con una **tabla de una página**: columnas = evidencia | figura/tabla donde aparece | conclusión | limitación. Esta tabla es literalmente la guía de calificación del profesor — no la omitas.

---

## Fase 8 — Presentación (máx. 8 diapositivas, 8 min + 4 de preguntas)

Sugerencia de estructura (1 diapositiva por punto, aprox.):
1. El problema y la pregunta central.
2. Datos y protocolo temporal.
3. A vs B — resultados.
4. Prueba de permutación (el momento más importante).
5. Segunda prueba de falsificación.
6. Apuesta C.
7. Decisión económica + recomendación.
8. Límites y próximos pasos.

Prepárense ambos para defender **cualquier** decisión técnica — en la sesión se elige una al azar de las que declaren en el README.

---

## Fase 9 — README.md

Debe incluir:
- Instrucciones de reproducción y versiones usadas.
- **Declaración de uso de IA**: para qué la usaron, qué verificaron ustedes mismos.
- **Tres decisiones técnicas importantes**: qué alternativas consideraron y qué evidencia inclinó la decisión final. (Se las pueden preguntar en la presentación — si no la pueden defender, no cuenta, sin importar quién escribió el código).
- Sección **"Candidato al Proyecto Final"**:
  - Qué modelo conservarían y dónde está el artefacto.
  - Quién usaría el puntaje y qué decisión tomaría con él.
  - Contrato preliminar de entrada/salida del modelo.
  - Límites, riesgos y datos que aún faltarían.

---

## Checklist de errores que cuestan puntos automáticamente

- [ ] ❌ Partición **aleatoria** en vez de temporal → −20
- [ ] ❌ Normalizar/seleccionar variables usando estadísticas del dataset **completo** (no solo train) → −15
- [ ] ❌ Reportar **exactitud** como métrica principal → −15
- [ ] ❌ Elegir arquitectura, umbral o apuesta **mirando el test set** → −10
- [ ] ❌ Afirmar que el orden aporta **sin** hacer la permutación controlada → −10

---

## Orden sugerido de trabajo (cronograma flexible)

| Semana | Enfoque |
|---|---|
| 1 | Fase 0-1: elegir ruta, construir/conseguir datos, partición temporal |
| 2 | Fase 2-3: Modelo A y Modelo B, comparación base |
| 3 | Fase 4-5: pruebas de falsificación + apuesta C |
| 4 | Fase 6-9: análisis económico, informe, presentación, README |

Ajusta según cuánto tiempo tengan hasta el 4 de septiembre.
