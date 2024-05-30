# Construye un chatbot LLM RAG con LangChain

Es probable que hayas interactuado con grandes modelos de lenguaje (LLM), como los que están detrás de ChatGPT de OpenAI, y hayas experimentado su notable capacidad para responder preguntas, resumir documentos, escribir código y mucho más. Si bien los LLM son notables por sí mismos, con un poco de conocimiento de programación, puedes aprovechar bibliotecas como [LangChain](https://python.langchain.com/docs/get_started/introduction) para crear tus propios chatbots con LLM que pueden hacer casi cualquier cosa.

En un entorno empresarial, una de las formas más populares de crear un chatbot impulsado por LLM es a través de la generación aumentada por recuperación (RAG). Cuando diseña un sistema RAG, utiliza un modelo de recuperación para recuperar información relevante, generalmente de una base de datos o corpus, y proporciona esta información recuperada a un LLM para generar respuestas contextualmente relevantes.

En este tutorial, te pondrás en la piel de un ingeniero de IA que trabaja para un gran sistema hospitalario. Construirá un chatbot RAG en LangChain que utiliza [Neo4j](https://neo4j.com/) para recuperar datos sobre los pacientes, las experiencias de los pacientes, las ubicaciones de los hospitales, las visitas, los pagadores de seguros y los médicos de su sistema hospitalario.

**En este taller, aprenderás a**:

- Usa **LangChain** para crear **chatbots** personalizados
- **Diseñe** un chatbot utilizando su comprensión de los requisitos comerciales y los datos del sistema hospitalario
- Trabajar con **bases de datos de gráficos**
- Configurar una instancia de **Neo4j** AuraDB
- Crea un chatbot **RAG** que recupere datos **estructurados** y **no estructurados** de Neo4j
- **Implementa** tu chatbot con **FastAPI** y **Streamlit**

Haga clic en el siguiente enlace para descargar el código fuente completo y los datos de este proyecto:

**Obtenga su código:** [Haga clic aquí para descargar el código fuente gratuito](https://realpython.com/bonus/build-llm-rag-chatbot-with-langchain-code/) de su chatbot LangChain.

## Demostración: Un chatbot de LLM RAG con LangChain y Neo4j

Al final de este tutorial, tendrás una [API REST](https://realpython.com/api-integration-in-python/) que sirve a tu chatbot LangChain. También tendrás una aplicación [Streamlit](https://streamlit.io/) que proporciona una buena interfaz de chat para interactuar con tu API:

Bajo el capó, la aplicación Streamlit envía sus mensajes a la API del chatbot, y el chatbot genera y envía una respuesta a la aplicación Streamlit, que se la muestra al usuario.

Obtendrá una visión general en profundidad de los datos a los que su chatbot tiene acceso más adelante, pero si está ansioso por probarlo, puede hacer preguntas similares a los ejemplos que se dan en la barra lateral:

Las preguntas de ejemplo se pueden encontrar en la barra lateral.

Aprenderás a abordar cada paso, desde la comprensión de los requisitos y los datos del negocio hasta la construcción de la aplicación Streamlit. Hay mucho que desempaquetar en este taller, pero no te sientas abrumado. Obetendrás algunos antecedentes sobre cada concepto introducido, junto con enlaces a fuentes externas que profundizarán tu comprensión. ¡Ahora es el momento de sumergirse!

## Requisitos previos

Este tutorial es el más adecuado para desarrolladores intermedios de Python que desean obtener experiencia práctica en la creación de chatbots personalizados. Además del conocimiento intermedio de Python, se beneficiará de tener una comprensión de alto nivel de los siguientes conceptos y tecnologías:

- Modelos de lenguaje grande (LLM) e [ingeniería rápida](https://realpython.com/practical-prompt-engineering/)
- [Acrustaciones de texto y bases de datos vectoriales](https://realpython.com/chromadb-vector-database/#represent-data-as-vectors)
- [Bases de datos gráficas](https://neo4j.com/developer/graph-database/) y [Neo4j](https://neo4j.com/docs/getting-started/languages-guides/neo4j-python/)
- [El ecosistema de desarrolladores de OpenAI](https://openai.com/product)
- [API REST](https://realpython.com/api-integration-in-python/) y [FastAPI](https://realpython.com/fastapi-python-web-apis/)
- [Programación asíncrona](https://realpython.com/async-io-python/)
- [Docker](https://realpython.com/tutorials/docker/) y [Docker Compose](https://docs.docker.com/compose/) 

Nada de lo mencionado anteriormente es un requisito previo difícil, así que no te preocupes si no te sientes bien informado en ninguno de ellos. Se te presentará cada concepto y tecnología a lo largo del camino. Además, no hay mejor manera de aprender estos requisitos previos que implementarlos usted mismo en este tutorial.

A continuación, obtendrás una breve descripción general del proyecto y comenzarás a aprender sobre LangChain.

## Descripción general del proyecto

A lo largo de este tutorial, crearás algunos directorios que componen tu chatbot final. Aquí hay un desglose de cada directorio:

- `langchain_intro/`te ayudará a familiarizarte con LangChain y a equiparte con las herramientas que necesitas para construir el chatbot que viste en la demostración, y no se incluirá en tu chatbot final. Cubrirás esto en el paso 1.

- `data/`tiene los datos sin procesar del sistema hospitalario almacenados como archivos CSV. Explorarás estos datos en el paso 2. En el paso 3, moverá estos datos a una base de datos Neo4j que su chatbot consultará para responder preguntas.

- `hospital_neo4j_etl/` contiene un script que carga los datos sin procesar de `data/` en su base de datos Neo4j. Tienes que ejecutar esto antes de construir tu chatbot, y aprenderás todo lo que necesitas saber sobre la configuración de una instancia de Neo4j en el paso 3.

- `chatbot_api/` es su aplicación [FastAPI](https://realpython.com/fastapi-python-web-apis/) aque sirve a su chatbot como punto final REST, y es el principal producto de este proyecto. Los subdirectorios`chatbot_api/src/agents/` y `chatbot_api/src/chains/` contienen los objetos LangChain que componen su chatbot. Aprenderá qué agentes y cadenas son más adelante, pero por ahora, solo sepa que su chatbot es en realidad un agente de LangChain compuesto de cadenas y funciones.

- `tests/`incluye dos scripts que prueban la rapidez con la que su chatbot puede responder a una serie de preguntas. Esto le dará una idea de cuánto tiempo ahorra al hacer solicitudes asíncronas a proveedores de LLM como OpenAI.

- `chatbot_frontend/`es tu aplicación Streamlit que interactúa con el punto final del chatbot inchatbot`chatbot_api/`. Esta es la interfaz de usuario que viste en la demostración, y la construirás en el paso 5.

Todas las variables de entorno necesarias para crear y ejecutar su chatbot se almacenarán en un archivo `.env`. Implementarás el código en `hospital_neo4j_etl/`, `chatbot_api` y `chatbot_frontend` como contenedores Docker que se orquestarán con Docker Compose. Si quieres experimentar con el chatbot antes de revisar el resto de este tutorial, puedes descargar los materiales y seguir las instrucciones del archivo README para que las cosas funcionen:

**Obtenga su código:** [Haga clic aquí para descargar el código fuente gratuito](https://realpython.com/bonus/build-llm-rag-chatbot-with-langchain-code/) de su chatbot LangChain.

Con la descripción general del proyecto y los requisitos previos detrás de ti, estás listo para comenzar con el primer paso: familiarizarte con LangChain.

## Paso 1: Familiarízate con LangChain

Antes de diseñar y desarrollar tu chatbot, necesitas saber cómo usar LangChain. En esta sección, conocerás los principales componentes y características de LangChain mediante la creación de una versión preliminar del chatbot de tu sistema hospitalario. Esto te dará todas las herramientas necesarias para construir tu chatbot completo.

Utilice su editor de código favorito para crear un nuevo proyecto de Python y asegúrese de crear un [entorno virtual](https://realpython.com/python-virtual-environments-a-primer/) para sus dependencias. Asegúrate de tener instalado Python 3.10 o posterior. Active su entorno virtual e instale las siguientes bibliotecas:

Shell

```
(venv) $ python -m pip install langchain==0.1.0 openai==1.7.2 langchain-openai==0.0.2 langchain-community==0.0.12 langchainhub==0.1.14
```

```shell
python -m pip install langchain==0.1.20 openai==1.28.1 langchain-openai==0.1.6 langchain-community==0.0.38 langchainhub==0.1.15
```



También querrás instalar [`python-dotenv`](https://pypi.org/project/python-dotenv/) para ayudarte a gestionar las variables del entorno:

Shell

```shell
(venv) $ python -m pip install python-dotenv
```

Python-dotenv carga variables de entorno de los archivos .env en tu entorno Python, y lo encontrarás útil a medida que desarrollas tu chatbot. Sin embargo, eventualmente implementará su chatbot con Docker, que puede manejar variables de entorno por usted, y ya no necesitará Python-dotenv.

Si aún no lo has hecho, tendrás que descargar  `reviews.csv`de los materiales o el repositorio de [GitHub](https://github.com/hfhoffman1144/langchain_neo4j_rag_app/blob/main/data/reviews.csv) para este taller:

**Obtenga su código:** [Haga clic aquí para descargar el código fuente gratuito](https://realpython.com/bonus/build-llm-rag-chatbot-with-langchain-code/) de su chatbot LangChain.

o puede descargar de la siguiente manera: 



```shell
wget https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/reviews.csv
```

A continuación, abra el directorio del proyecto y agregue las siguientes carpetas y archivos:

```
./
│
├── data/
│   └── reviews.csv
│
├── langchain_intro/
│   ├── chatbot.py
│   ├── create_retriever.py
│   └── tools.py
│
└── .env
```

El archivo `reviews.csv` en `data/` es el que acabas de descargar, y los archivos restantes que veas deberían estar vacíos.

¡Ya estás listo para empezar a construir tu primer chatbot con LangChain!

### Modelos de chat

Es posible que hayas adivinado que el componente principal de LangChain es el [LLM](https://python.langchain.com/docs/modules/model_io/llms/). LangChain proporciona una interfaz modular para trabajar con proveedores de LLM como OpenAI, Cohere, HuggingFace, Anthropic, Together AI y otros. En la mayoría de los casos, todo lo que necesita es una clave de API del proveedor de LLM para comenzar a usar el LLM con LangChain. LangChain también es compatible con LLM u otros modelos de idioma alojados en su propia máquina.

Utilizarás OpenAI para este tutorial, pero ten en cuenta que hay muchos grandes proveedores de código abierto y cerrado por ahí. Siempre puede probar diferentes proveedores y optimizarlos en función de las necesidades de su aplicación y las limitaciones de costes. Antes de seguir adelante, asegúrese de que está registrado para una cuenta de OpenAI y de que tiene una [clave API](https://openai.com/product) válida.

Una vez que tenga su clave de la API de OpenAI, agréguela a su archivo `.env`:

`OPENAI_API_KEY=<YOUR-OPENAI-API-KEY>`

Si bien puedes interactuar directamente con los objetos LLM en LangChain, una abstracción más común es el [modelo de chat](https://python.langchain.com/docs/modules/model_io/chat/). Los modelos de chat utilizan LLM bajo el capó, pero están diseñados para conversaciones e interactúan con [los mensajes de chat](https://python.langchain.com/docs/modules/model_io/chat/quick_start#messages) en lugar de con el texto sin procesar.

Usando los mensajes de chat, usted proporciona un LLM con detalles adicionales sobre el tipo de mensaje que está enviando. Todos los mensajes tienen propiedades de `rol` y `contenido`. El `rol` le dice al LLM quién está enviando el mensaje, y el `contenido` es el mensaje en sí. Estos son los mensajes más utilizados:

- `HumanMessage`: Un mensaje del usuario que interactúa con el modelo de idioma.
- `AIMessage`: Un mensaje del modelo de idioma.
- `SystemMessage`: Un mensaje que le dice al modelo lingüístico cómo comportarse. No todos los proveedores admiten `SystemMessage`.

Hay otros tipos de mensajes, como `FunctionMessage` y `ToolMessage`, pero aprenderás más sobre ellos cuando construyas un [agent](https://python.langchain.com/docs/modules/agents/).

Empezar con los modelos de chat en LangChain es sencillo. Para crear una instancia de un modelo de chat de OpenAI, vaya a `langchain_intro` y agregue el siguiente código a `chatbot.py`:

Python `langchain_intro/chatbot.py`

```
import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
```

Primero importas `dotenv` y `ChatOpenAI`. Luego llamas a `dotenv.load_dotenv()` que lee y almacena variables de entorno de `.env`. De forma predeterminada, `dotenv.load_dotenv()` asume que `.env` se encuentra en el directorio de trabajo actual, pero puede pasar la ruta a otros directorios si `.env` se encuentra en otro lugar.

A continuación, crea una instancia de un modelo de `ChatOpenAI` utilizando [GPT 3.5 Turbo](https://platform.openai.com/docs/models/gpt-3-5) como LLM base, y establece la `temperatura` en 0. OpenAI ofrece una diversidad de [models](https://platform.openai.com/docs/models) con diferentes precios, capacidades y rendimientos. GPT 3.5 turbo es un gran modelo para empezar porque funciona bien en muchos casos de uso y es más barato que los modelos más recientes como GPT 4 y más allá.



> **Nota**: Es una idea errónea común que establecer la `temperature=0` garantiza respuestas deterministas de los modelos GPT. Si bien las respuestas están más cerca de las deterministas cuando `temperature=0`, [no hay garantía](https://arxiv.org/abs/2308.02828) de que obtengas la misma respuesta para solicitudes idénticas. Debido a esto, los modelos GPT podrían dar resultados ligeramente diferentes a los que se ven en los ejemplos a lo largo de este tutorial.

Para usar `chat_model`, abra el directorio del proyecto, inicie un intérprete de Python y ejecute el siguiente código:

Shell Python

```shell
>>> from langchain.schema.messages import HumanMessage, SystemMessage
>>> from langchain_intro.chatbot import chat_model

>>> messages = [
...     SystemMessage(
...     content="""Eres un asistente con conocimientos sobre
        Atención médica. Solo responda a las preguntas 
        relacionadas con la atención médica."""
...     ),
...     HumanMessage(content="¿Qué es la atención administrada de Medicaid?"),
... ]
>>> chat_model.invoke(messages)

AIMessage(content='La atención administrada de Medicaid es un modelo de prestación de servicios de atención médica en el que los beneficiarios de Medicaid reciben sus servicios a través de organizaciones
de atención administrada. Estas organizaciones son responsables de coordinar y proporcionar los servicios de atención médica a los beneficiarios, con el objetivo de mejorar la calidad de la atención, cont
rolar los costos y mejorar los resultados de salud. Los beneficiarios de Medicaid que están inscritos en un plan de atención administrada suelen tener un médico de atención primaria que coordina su atenci
ón y los refiere a especialistas según sea necesario.', response_metadata={'token_usage': {'completion_tokens': 130, 'prompt_tokens': 54, 'total_tokens': 184}, 'model_name': 'gpt-3.5-turbo-0125', 'system_
fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-082e35bd-94f9-4c87-8a38-ea9df731b52c-0')
```

En este bloque, importas `HumanMessage` y `SystemMessage`, así como tu modelo de chat. A continuación, define una lista con un `SystemMessage` y un `HumanMessage` y los ejecuta a través de `chat_model` con `chat_model.invoke()`. Bajo el capó, `chat_model` hace una solicitud a un punto final de OpenAI que sirve a `gpt-3.5-turbo-0125`, y los resultados se devuelven como un `AIMessage`.

> **Nota**:
> 
> Puede que copiar y pegar código multilínea de este tutorial en su [REPL estándar de Python](https://realpython.com/python-repl/) es un poco engorroso. Para una mejor experiencia, podría instalar un REPL alternativo de Python, como [IPython](https://realpython.com/ipython-interactive-python-shell/), [bpython](https://realpython.com/bpython-alternative-python-repl/) o [ptpython](https://realpython.com/ptpython-shell/), en su entorno virtual y ejecutar las interacciones REPL con ellos.

Como puedes ver, el modelo de chat respondió ¿Qué es la atención administrada por Medicaid? Proporcionado en el `HumanMessage`. Es posible que te preguntes qué hizo el modelo de chat con `SystemMessage` en este contexto. Fíjate en lo que sucede cuando haces la siguiente pregunta:

Shell Python

```shell
messages = [
    SystemMessage(
        content="""Eres un asistente con conocimientos sobre
        Atención médica. Solo responda a las preguntas relacionadas 
        con la atención médica."""
    ),
    HumanMessage(content="¿Cómo cambio un neumático?"),
]
>>> chat_model.invoke(messages)



AIMessage(content='Lo siento, pero cambiar un neumático no está relacionado con la atención médica. ¿Hay algo más en lo que pueda ayudarte en relación con la salud o la atención médica?', response_metadat
a={'token_usage': {'completion_tokens': 41, 'prompt_tokens': 56, 'total_tokens': 97}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-5d
30c241-53bf-453d-ac54-ebf18c44d9bc-0')

```

Como se describió anteriormente, el `SystemMessage` le dice al modelo cómo comportarse. En este caso, le dijiste al modelo *que solo respondiera a preguntas relacionadas con la atención médica*. Es por eso que se niega a decirte cómo cambiar tu neumático. La capacidad de controlar cómo se relaciona un LLM con el usuario a través de instrucciones de texto es poderosa, y esta es la base para crear chatbots personalizados a través de la [prompt engineering](https://realpython.com/practical-prompt-engineering/).

Si bien los mensajes de chat son una buena abstracción y son buenos para asegurarse de que le está dando al LLM el tipo correcto de mensaje, también puede pasar cadenas sin procesar a los modelos de chat:

Shell Python

```
>>> chat_model.invoke("Que es la presión arterial?")
AIMessage(content='Blood pressure is the force exerted by
the blood against the walls of the blood vessels, particularly
the arteries, as it is pumped by the heart. It is measured in
millimeters of mercury (mmHg) and is typically expressed as two
numbers: systolic pressure over diastolic pressure. The systolic
pressure represents the force when the heart contracts and pumps
blood into the arteries, while the diastolic pressure represents
the force when the heart is at rest between beats. Blood pressure
is an important indicator of cardiovascular health and can be influenced
by various factors such as age, genetics, lifestyle, and underlying medical
conditions.')
```

En este bloque de código, pasas la cadena *¿Qué es la presión arterial?* Directamente a `chat_model.invoke()`. Si desea controlar el comportamiento de LLM sin un `SystemMessage` aquí, puede incluir instrucciones en la entrada de cadena.

> **Nota**: En estos ejemplos, usaste `.invoke()`, pero LangChain tiene  [otros métodos](https://python.langchain.com/docs/expression_language/interface) que interactúan con los LLM. Por ejemplo, `.stream()` devuelve la respuesta un token a la vez, y `.batch()` acepta una lista de mensajes a los que el LLM responde en una llamada.

Cada método también tiene un método asíncrono análogo. Por ejemplo, puedes ejecutar `.invoke()` de forma asíncrona con `ainvoke()`. 

A continuación, aprenderá una forma modular de guiar la respuesta de su modelo, como lo hizo con el `SystemMessage`, lo que facilita la personalización de su chatbot.

### Plantillas rápidas

LangChain le permite diseñar indicaciones modulares para su chatbot con [plantillas de solicitudes](https://python.langchain.com/docs/modules/model_io/prompts/quick_start). Citando la documentación de LangChain, puede pensar en plantillas de indicaciones como *recetas predefinidas para generar indicaciones para modelos de lenguaje*.

Supongamos que quieres crear un chatbot que responda a las preguntas sobre las experiencias de los pacientes a partir de sus reseñas. Así es como podría ser una plantilla de aviso para esto:

Shell Python

```shell
from langchain.prompts import ChatPromptTemplate

review_template_str = """Su trabajo consiste en utilizar 
revisiones de pacientes para responder preguntas sobre su experiencia 
en un hospital. Utilice el siguiente contexto para responder preguntas. 
Sea lo más detallado posible, pero no invente ninguna información que
no provenga del contexto. Si no sabes una respuesta, di que no sabes.

{context}

{question}
"""

review_template = ChatPromptTemplate.from_template(review_template_str)

context = "¡Tuve una estancia estupenda!"
question = "¿Alguien tuvo una experiencia positiva?"

review_template.format(context=context, question=question)
```

```
'Human: Su trabajo consiste en utilizar \nrevisiones de pacientes para responder preguntas sobre su experiencia \nen un hospital. Utilice el siguiente contexto para responder preguntas. \nSea lo más detal
lado posible, pero no invente ninguna información que\nno provenga del contexto. Si no sabes una respuesta, di que no sabes.\n\n¡Tuve una estancia estupenda!\n\n¿Alguien tuvo una experiencia positiva?\n'
```

Primero importa `ChatPromptTemplate` y define `review_template_str`, que contiene las instrucciones que pasará al modelo, junto con el `context` de las variables y la `question` en los [reemplazo  fields](https://realpython.com/python-f-strings/#the-strformat-method) que LangChain delimita con llaves (`{}`). A continuación, crea un objeto `ChatPromptTemplate` desde `review_template_str` utilizando el [class method](https://realpython.com/instance-class-and-static-methods-demystified/) `.from_template()`.

Con `review_template` instanciado, puede pasar el `context` y la `question` a la plantilla de cadena con `review_template.format()`. Los resultados pueden parecer que no has hecho nada más que la [interpolación de cadenas estándar de Python]([Python&#x27;s F-String for String Interpolation and Formatting – Real Python](https://realpython.com/python-f-strings/)), pero las plantillas de prompt tienen muchas características útiles que les permiten integrarse con modelos de chat.

Observe cómo su llamada anterior a `review_template.format()` generó una cadena con *Human* al principio. Esto se debe a que `ChatPromptTemplate.from_template()` asume que la plantilla de cadena es un mensaje humano de forma predeterminada. Para cambiar esto, puede crear plantillas de aviso más detalladas para cada mensaje de chat que desee que el modelo procese:

```shell



review_system_template_str = """Su trabajo consiste en utilizar 
reseñas de pacientes para responder preguntas sobre su 
experiencia en un hospital. Utilice el siguiente contexto 
para responder preguntas. Sea lo más detallado posible, 
pero no invente ninguna información que no provenga del 
contexto. Si no sabe una respuesta, diga que no la sabe.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=review_system_template_str
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"], template="{question}"
    )
)

messages = [review_system_prompt, review_human_prompt]
review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)
context = "¡Tuve una estancia estupenda!"
question = "¿Alguien tuvo una experiencia positiva?"

review_prompt_template.format_messages(context=context, question=question)
```

En este bloque, importas plantillas de aviso separadas para `HumanMessage` y `SystemMessage`. A continuación, define una cadena, `review_system_template_str`, que sirve como plantilla para un `SystemMessage`. Observe cómo solo declara una variable de `context` en `review_system_template_str`.

A partir de esto, creas `review_system_prompt`, que es una plantilla de aviso específicamente para `SystemMessage`. A continuación, creas un `review_human_prompt` para el` HumanMessage`. Observe cómo el parámetro de la `template` es solo una cadena con la variable de `question`.

A continuación, agrega `review_system_prompt` y `review_human_prompt` a una lista llamada `messages` y crea `review_prompt_template`, que es el objeto final que abarca las plantillas de aviso tanto para `SystemMessage` como para `HumanMessage`. Llamar a `review_prompt_template.format_messages(context=context, question=question)` genera una lista con un `SystemMessage` y `HumanMessage`, que se puede pasar a un modelo de chat.

Para ver cómo combinar modelos de chat y plantillas de indicaciones, construirás una cadena con el lenguaje de expresión LangChain (LCEL). Esto le ayuda a desbloquear la funcionalidad principal de LangChain de crear interfaces modulares personalizadas sobre modelos de chat.

### Cadenas y lenguaje de expresión LangChain (LCEL)

El pegamento que conecta los modelos de chat, las indicaciones y otros objetos en LangChain es la [cadena](https://python.langchain.com/docs/modules/chains). Una cadena no es más que una secuencia de llamadas entre objetos en LangChain. La forma recomendada de construir cadenas es usar el [lenguaje de expresión LangChain (LCEL).](https://python.langchain.com/docs/expression_language/)

Para ver cómo funciona esto, eche un vistazo a cómo crearía una cadena con un modelo de chat y una plantilla de aviso:

Python langchain_intro/chatbot.py

```python:
import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
 PromptTemplate,
 SystemMessagePromptTemplate,
 HumanMessagePromptTemplate,
 ChatPromptTemplate,
)

dotenv.load_dotenv()

review_template_str = """Su trabajo consiste en utilizar 
reseñas de pacientes para responder preguntas sobre su experiencia
 en un hospital. Utilice el siguiente contexto para responder preguntas. 
Sea lo más detallado posible, pero no invente ninguna información que 
no provenga del contexto. Si no sabe una respuesta, diga que no la sabe.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
 prompt=PromptTemplate(
 input_variables=["context"],
 template=review_template_str,
 )
)

review_human_prompt = HumanMessagePromptTemplate(
 prompt=PromptTemplate(
 input_variables=["question"],
 template="{question}",
 )
)
messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
 input_variables=["context", "question"],
 messages=messages,
)

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

review_chain = review_prompt_template | chat_model
```

Las líneas 1 a 42 son lo que ya has hecho. Es decir, usted define `review_prompt_template`, que es una plantilla rápida para responder preguntas sobre las revisiones de los pacientes, y crea una instancia de un modelo de chat `gpt-3.5-turbo-0125`. En la línea 44, se define `review_chain` con el símbolo `|`, que se utiliza para encadenar `review_prompt_template` y `chat_model`.

Esto crea un objeto, `review_chain`, que puede pasar preguntas a través de `review_prompt_template` y `chat_model` en una sola llamada a la función. En esencia, esto abstracta todos los detalles internos de `review_chain`, lo que le permite interactuar con la cadena como si fuera un modelo de chat.

Después de guardar el `chatbot.py` actualizado, inicie una nueva sesión de REPL en la carpeta de su proyecto base. Así es como puedes usar `review_chain`:

Python Shell

```
from langchain_intro.chatbot import review_chain

context = "I had a great stay!"
question = "Did anyone have a positive experience?"

review_chain.invoke({"context": context, "question": question})
```

En este bloque, importas `review_chain` y defines el `context` y la `questio` como antes. A continuación, pasas un diccionario con el `context` de las claves y la `question` a `review_chan.invoke()`. Esto pasa el `context` y la pregunta a través de la plantilla de prompt y el modelo de chat para generar una respuesta.

> **Nota**: Al llamar a las cadenas, puedes usar todos los mismos [métodos](https://python.langchain.com/docs/expression_language/interface) que admite un modelo de chat.

En general, el LCEL le permite crear cadenas de longitud arbitraria con [el símbolo de tubería](https://en.wikipedia.org/wiki/Vertical_bar#Pipe) (`|`). Por ejemplo, si querías dar formato a la respuesta del modelo, entonces podrías añadir un [analizador de salida](https://python.langchain.com/docs/modules/model_io/output_parsers/) a la cadena:

Python `langchain_intro/chatbot.py`

```
import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

# ...

output_parser = StrOutputParser()

review_chain = review_prompt_template | chat_model | output_parser
```

Aquí, agrega una instancia de `StrOutputParser()` a `review_chain`, lo que hará que la respuesta del modelo sea más legible. Inicia una nueva sesión de REPL y pruébalo:

Shell

```
from langchain_intro.chatbot import review_chain

context = "¡Tuve una estancia estupenda!"
question = "¿Alguien tuvo una experiencia positiva?"

review_chain.invoke({"context": context, "question": question})
```

```
'Sí, el paciente tuvo una gran estancia y tuvo una
experiencia positiva en el hospital.'
```

Este bloque es el mismo que antes, excepto que ahora puedes ver que `review_chain` devuelve una cadena con un buen formato en lugar de un `AIMessage`.

El poder de las cadenas está en la creatividad y flexibilidad que te ofrecen. Puedes encadenar pipelines complejas para crear tu chatbot, y terminas con un objeto que ejecuta tu canalización en una sola llamada de método. A continuación, colocarás otro objeto en `review_chain` para recuperar documentos de una base de datos vectorial.

### Objetos de recuperación

El objetivo de `review_chain` es responder a las preguntas sobre las experiencias de los pacientes en el hospital a partir de sus revisiones. Hasta ahora, has aprobado manualmente las revisiones como contexto para la pregunta. Aunque esto puede funcionar para un pequeño número de revisiones, no se escala bien. Además, incluso si puede encajar todas las revisiones en la ventana de contexto del modelo, no hay garantía de que utilice las revisiones correctas al responder a una pregunta.

Para superar esto, necesitas un [retriever](https://python.langchain.com/docs/modules/data_connection/). El proceso de recuperar documentos relevantes y pasarlos a un modelo de idioma para responder preguntas se conoce como [generación aumentada por recuperación (RAG).](https://en.wikipedia.org/wiki/Prompt_engineering#Retrieval-augmented_generation)

Para este ejemplo, almacenarás todas las revisiones en una [base de datos vectorial](https://en.wikipedia.org/wiki/Vector_database) llamada [ChromaDB](https://www.trychroma.com/). Si no estás familiarizado con esta herramienta y los temas de base de datos, echa un vistazo a [Incrustaciones y bases de datos vectoriales con ChromaDB](https://realpython.com/chromadb-vector-database/) antes de continuar.

Puedes instalar ChromaDB con el siguiente comando:

Shell

`(venv) $ python -m pip install chromadb==0.4.22`

Con esto instalado, puede usar el siguiente código para crear una base de datos vectorial ChromaDB con reseñas de pacientes:

Python `langchain_intro/create_retriever.py`

```
import dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

reviews_vector_db = Chroma.from_documents(
    reviews, OpenAIEmbeddings(), persist_directory=REVIEWS_CHROMA_PATH
)
```

En las líneas 2 a 4, se importan las dependencias necesarias para crear la base de datos vectorial. A continuación, define `REVIEWS_CSV_PATH` y `REVIEWS_CHROMA_PATH`, que son rutas donde se almacenan los datos de las revisiones sin procesar y donde la base de datos vectorial almacenará los datos, respectivamente.

Más tarde obtendrás una visión general de los datos del sistema hospitalario, pero todo lo que necesitas saber por ahora es que `reviews.csv` almacena las reseñas de los pacientes. La columna de `revisión` en `reviews.csv` es una cadena con la revisión del paciente.

En las líneas 11 y 12, se cargan las reseñas utilizando el `CSVLoader` de LangChain. En las líneas 14 a 16, se crea una instancia de ChromaDB a partir de `reviews` utilizando el modelo de embedding  OpenAI predeterminado, y se almacenan las incrustaciones de revisión en `REVIEWS_CHROMA_PATH`.

> **Nota**: En la práctica, si está incrustando un documento grande, debe usar un [divisor de texto](https://python.langchain.com/docs/modules/data_connection/document_transformers/). Los divisores de texto dividen el documento en trozos más pequeños antes de ejecutarlos a través de un modelo de incrustación. Esto es importante porque los modelos de incrustación tienen una ventana de contexto de tamaño fijo, y a medida que crece el tamaño del texto, la capacidad de una incrustación para representar con precisión el texto disminuye.

Para este ejemplo, puedes incrustar cada revisión individualmente porque son relativamente pequeñas.

A continuación, abra un terminal y ejecute el siguiente comando desde el directorio del proyecto:

Shell

`(venv) $ python langchain_intro/create_retriever.py`

Solo debería tardar un minuto más o menos en ejecutarse, y después puedes comenzar a realizar una búsqueda semántica sobre las incrustaciones de revisión:

Shell Python

```
>>> import dotenv
>>> from langchain_community.vectorstores import Chroma
>>> from langchain_openai import OpenAIEmbeddings

>>> REVIEWS_CHROMA_PATH = "chroma_data/"

>>> dotenv.load_dotenv()
True

>>> reviews_vector_db = Chroma(
...     persist_directory=REVIEWS_CHROMA_PATH,
...     embedding_function=OpenAIEmbeddings(),
... )

>>> question = """Has anyone complained about
...            communication with the hospital staff?"""
>>> relevant_docs = reviews_vector_db.similarity_search(question, k=3)

>>> relevant_docs[0].page_content
'review_id: 73\nvisit_id: 7696\nreview: I had a frustrating experience
at the hospital. The communication between the medical staff and me was
unclear, leading to misunderstandings about my treatment plan. Improvement
is needed in this area.\nphysician_name: Maria Thompson\nhospital_name:
Little-Spencer\npatient_name: Terri Smith'

>>> relevant_docs[1].page_content
'review_id: 521\nvisit_id: 631\nreview: I had a challenging time at the
hospital. The medical care was adequate, but the lack of communication
between the staff and me left me feeling frustrated and confused about my
treatment plan.\nphysician_name: Samantha Mendez\nhospital_name:
Richardson-Powell\npatient_name: Kurt Gordon'

>>> relevant_docs[2].page_content
'review_id: 785\nvisit_id: 2593\nreview: My stay at the hospital was challenging.
The medical care was adequate, but the lack of communication from the staff
created some frustration.\nphysician_name: Brittany Harris\nhospital_name:
Jones, Taylor and Garcia\npatient_name: Ryan Jacobs'
```

Importa las dependencias necesarias para llamar a ChromaDB y especifica la ruta de acceso a los datos almacenados de ChromaDB en `REVIEWS_CHROMA_PATH`. A continuación, carga variables de entorno usando `dotenv.load_dotenv()` y crea una nueva instancia de `Chroma` que apunta a su base de datos vectorial. Observe cómo tiene que volver a especificar una función de incrustación cuando se conecte a su base de datos vectorial. Asegúrate de que esta es la misma función de incrustación que usaste para crear las incrustaciones.

A continuación, define una pregunta y llama a `.similarity_search()` en `reviews_vector_db`, pasando la `question` y `k=3`. Esto crea una incrustación para la pregunta y busca en la base de datos vectorial las tres incrustaciones de revisión más similares a la incrustación de preguntas. En este caso, ves tres reseñas en las que los pacientes se quejaron de la comunicación, ¡que es exactamente lo que pediste!

Lo último que puede hacer es agregar su retriever de reseñas a `review_chain` para que las revisiones relevantes se pasen al mensaje como contexto. Así es como lo haces:

Python `langchain_intro/chatbot.py`

```
import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough

REVIEWS_CHROMA_PATH = "chroma_data/"

# ...

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=OpenAIEmbeddings()
)

reviews_retriever  = reviews_vector_db.as_retriever(k=10)

review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)
```

Como antes, importa las dependencias de ChromaDB, especifica la ruta de acceso a sus datos de ChromaDB y crea una instancia de un nuevo objeto `Chroma`. A continuación, crea `reviews_retriever` llamando a `.as_retriever()` en `reviews_vector_db` para crear un objeto retriever que añadirá a `review_chain`. Debido a que especificó `k=10`, el retriever obtendrá las diez reseñas más similares a la pregunta del usuario.

A continuación, añades un diccionario con claves de `context` y preguntas en la parte delantera de `review_chain`. En lugar de pasar el `context` manualmente, `review_chain` pasará su pregunta al retriever para extraer las revisiones relevantes. La asignación de `question` a un objeto `RunnablePassthrough` garantiza que la pregunta se pase sin cambios al siguiente paso de la cadena.

Ahora tiene una cadena totalmente funcional que puede responder a las preguntas sobre las experiencias de los pacientes a partir de sus revisiones. Inicia una nueva sesión de REPL y pruébela:

```
>>> from langchain_intro.chatbot import review_chain

>>> question = """Has anyone complained about
...            communication with the hospital staff?"""
>>> review_chain.invoke(question)
'Yes, several patients have complained about communication
with the hospital staff. Terri Smith mentioned that the
communication between the medical staff and her was unclear,
leading to misunderstandings about her treatment plan.
Kurt Gordon also mentioned that the lack of communication
between the staff and him left him feeling frustrated and
confused about his treatment plan. Ryan Jacobs also experienced
frustration due to the lack of communication from the staff.
Shannon Williams also mentioned that the lack of communication
between the staff and her made her stay at the hospital less enjoyable.'
```

Como puede ver, solo llama a `review_chain.invoke(pregunta)` para obtener respuestas aumentadas por la recuperación sobre las experiencias de los pacientes de sus revisiones. Mejorarás esta cadena más tarde almacenando incrustaciones de revisión, junto con otros metadatos, en Neo4j.

Ahora que entiendes los modelos de chat, las indicaciones, las cadenas y la recuperación, estás listo para sumergirte en el último concepto de LangChain: los agentes.

### Agentes

Hasta ahora, has creado una cadena para responder preguntas utilizando las reseñas de los pacientes. ¿Qué pasa si quieres que tu chatbot también responda a preguntas sobre otros datos del hospital, como los tiempos de espera del hospital? Idealmente, su chatbot puede cambiar sin problemas entre responder a la revisión del paciente y las preguntas sobre el tiempo de espera, dependiendo de la consulta del usuario. Para lograr esto, necesitarás los siguientes componentes:

1. La cadena de revisión de pacientes que ya has creado
2. Una función que puede buscar los tiempos de espera en un hospital
3. Una forma de que un LLM sepa cuándo debe responder preguntas sobre las experiencias de los pacientes o buscar los tiempos de espera

Para lograr la tercera capacidad, necesitas un [agente](https://python.langchain.com/docs/modules/agents/).

Un agente es un modelo de lenguaje que decide una secuencia de acciones a ejecutar. A diferencia de las cadenas en las que la secuencia de acciones está codificada, los agentes utilizan un modelo de lenguaje para determinar qué acciones tomar y en qué orden.

Antes de construir el agente, cree la siguiente función para generar tiempos de espera falsos para un hospital:

Python `langchain_intro/tools.py`

```
import random
import time

def get_current_wait_time(hospital: str) -> int | str:
    """Dummy function to generate fake wait times"""

    if hospital not in ["A", "B", "C", "D"]:
        return f"Hospital {hospital} does not exist"

    # Simulate API call delay
    time.sleep(1)

    return random.randint(0, 10000)
```

En `get_current_wait_time()`, pasas el nombre de un hospital, compruebas si es válido y luego generas un número aleatorio para simular un tiempo de espera. En realidad, esto sería algún tipo de consulta de base de datos o llamada a la API, pero esto servirá para el mismo propósito para esta demostración.

Ahora puedes crear un agente que decida entre `get_current_wait_time()` y `review_chain.invoke()` dependiendo de la pregunta:

Python `langchain_intro/chatbot.py`

```
import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub
from langchain_intro.tools import get_current_wait_time

# ...

tools = [
    Tool(
        name="Reviews",
        func=review_chain.invoke,
        description="""Useful when you need to answer questions
        about patient reviews or experiences at the hospital.
        Not useful for answering questions about specific visit
        details such as payer, billing, treatment, diagnosis,
        chief complaint, hospital, or physician information.
        Pass the entire question as input to the tool. For instance,
        if the question is "What do patients think about the triage system?",
        the input should be "What do patients think about the triage system?"
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_time,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. This tool returns wait times in
        minutes. Do not pass the word "hospital" as input,
        only the hospital name itself. For instance, if the question is
        "What is the wait time at hospital A?", the input should be "A".
        """,
    ),
]

hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

agent_chat_model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0,
)

hospital_agent = create_openai_functions_agent(
    llm=agent_chat_model,
    prompt=hospital_agent_prompt,
    tools=tools,
)

hospital_agent_executor = AgentExecutor(
    agent=hospital_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
```

En este bloque, importa algunas dependencias adicionales que necesitarás para crear el agente. A continuación, define una lista de objetos de `Tool`. Una `Tool` es una interfaz que un agente utiliza para interactuar con una función. Por ejemplo, la primera herramienta se llama `Reviews` y llama a `review_chain.invoke()` si la pregunta cumple con los criterios de `description`.

Observe cómo la `description` da instrucciones al agente sobre cuándo debe llamar a la herramienta. Aquí es donde las buenas habilidades de ingeniería rápidas son primordiales para garantizar que el LLM llame a la herramienta correcta con las entradas correctas.

La segunda `Tool` en las `Tools` se llama `Waits`, y llama a `get_current_wait_time()`. Una vez más, el agente tiene que saber cuándo usar la herramienta `Waits` y qué entradas pasar a ella dependiendo de la `description`.

A continuación, inicializa un objeto `ChatOpenAI` usando **gpt-3.5-turbo-1106** como modelo de idioma. A continuación, crea un agente de funciones OpenAI con `create_openai_functions_agent()`. Esto crea un agente diseñado para pasar entradas a las funciones. Lo hace devolviendo objetos JSON válidos que almacenan entradas de función y su valor correspondiente.

Para crear el tiempo de ejecución del agente, pasas el agente y las herramientas a `AgentExecutor`. Establecer `return_intermediate_steps` y `verbose` en `True` le permitirá ver el proceso de pensamiento del agente y las herramientas a las que llama.

Inicia una nueva sesión de REPL para darle una vuelta a tu nuevo agente:

```
>>> from langchain_intro.chatbot import hospital_agent_executor

>>> hospital_agent_executor.invoke(
...     {"input": "What is the current wait time at hospital C?"}
... )

> Entering new AgentExecutor chain...

Invoking: `Waits` with `C`

1374The current wait time at Hospital C is 1374 minutes.

> Finished chain.
{'input': 'What is the current wait time at hospital C?',
'output': 'The current wait time at Hospital C is 1374 minutes.',
'intermediate_steps': [(AgentActionMessageLog(tool='Waits',
tool_input='C', log='\nInvoking: `Waits` with `C`\n\n\n',
message_log=[AIMessage(content='', additional_kwargs={'function_call':
{'arguments': '{"__arg1":"C"}', 'name': 'Waits'}})]), 1374)]}

>>> hospital_agent_executor.invoke(
...     {"input": "What have patients said about their comfort at the hospital?"}
... )

> Entering new AgentExecutor chain...

Invoking: `Reviews` with `What have patients said about their comfort at the
hospital?`

Patients have mentioned both positive and negative aspects of their comfort at
the hospital. One patient mentioned that the hospital's dedication to patient
comfort was evident in the well-designed private rooms and comfortable furnishings,
which made their recovery more bearable and contributed to an overall positive
experience. However, other patients mentioned that the uncomfortable beds made
it difficult for them to get a good night's sleep during their stay, affecting
their overall comfort. Another patient mentioned that the outdated and
uncomfortable beds affected their overall comfort, despite the doctors being
knowledgeable and the hospital having a clean environment. Patients have shared
mixed feedback about their comfort at the hospital. Some have praised the well-designed
private rooms and comfortable furnishings, which contributed to a positive experience.
However, others have mentioned discomfort due to the outdated and uncomfortable beds,
affecting their overall comfort despite the hospital's clean environment and knowledgeable
doctors.

> Finished chain.
{'input': 'What have patients said about their comfort at the hospital?', 'output':
"Patients have shared mixed feedback about their comfort at the hospital. Some have
praised the well-designed private rooms and comfortable furnishings, which contributed
to a positive experience. However, others have mentioned discomfort due to the outdated
and uncomfortable beds, affecting their overall comfort despite the hospital's clean
environment and knowledgeable doctors.", 'intermediate_steps':
[(AgentActionMessageLog(tool='Reviews', tool_input='What have patients said about their
comfort at the hospital?', log='\nInvoking: `Reviews` with `What have patients said about
their comfort at the hospital?`\n\n\n', message_log=[AIMessage(content='',
additional_kwargs={'function_call': {'arguments': '{"__arg1":"What have patients said about
their comfort at the hospital?"}', 'name': 'Reviews'}})]), "Patients have mentioned both
positive and negative aspects of their comfort at the hospital. One patient mentioned that
the hospital's dedication to patient comfort was evident in the well-designed private rooms
and comfortable furnishings, which made their recovery more bearable and contributed to an
overall positive experience. However, other patients mentioned that the uncomfortable beds
made it difficult for them to get a good night's sleep during their stay, affecting their
overall comfort. Another patient mentioned that the outdated and uncomfortable beds affected
their overall comfort, despite the doctors being knowledgeable and the hospital having a clean
environment.")]}
```

Primero importa el agente y luego llama a `hospital_agent_executor.invoke()` con una pregunta sobre el tiempo de espera. Como se indica en la salida, el agente sabe que está preguntando por un tiempo de espera, y pasa `C` como entrada a la herramienta `Waits`. A continuación, la herramienta `Waits` llama a `get_current_wait_time(hospital="C")` y devuelve el tiempo de espera correspondiente al agente. A continuación, el agente utiliza este tiempo de espera para generar su resultado final.

Un proceso similar sucede cuando le preguntas al agente sobre las revisiones de la experiencia del paciente, excepto que esta vez el agente sabe que debe llamar a la herramienta de `Reviews` con 

*¿Qué han dicho los pacientes sobre su comodidad en el hospital?* Como entrada. La herramienta `Reviews` ejecuta `review_chain.invoke()` usando su pregunta completa como entrada, y el agente utiliza la respuesta para generar su salida.

Esta es una capacidad profunda. Los agentes dan a los modelos de lenguaje la capacidad de realizar casi cualquier tarea para la que puedas escribir código. Imagina todos los chatbots increíbles y potencialmente peligrosos que podrías construir con agentes.

Ahora tienes todos los requisitos previos de LangChain necesarios para construir un chatbot personalizado. A continuación, te pondrás tu sombrero de ingeniero de IA y aprenderás sobre los requisitos comerciales y los datos necesarios para construir tu chatbot del sistema hospitalario.

Todo el código que has escrito hasta ahora tenía la intención de enseñarte los fundamentos de LangChain, y no se incluirá en tu chatbot final. Siéntase libre de comenzar con un directorio vacío en el paso 2, donde comenzará a construir su chatbot.

## Paso 2: Comprender los requisitos y datos del negocio

Antes de empezar a trabajar en cualquier proyecto de IA, necesitas entender el problema que quieres resolver y hacer un plan de cómo lo vas a resolver. Esto implica definir claramente el problema, recopilar requisitos, comprender los datos y la tecnología disponibles para usted y establecer expectativas claras con las partes interesadas. Para este proyecto, comenzarás por definir el problema y recopilar los requisitos comerciales para tu chatbot.

### Comprender el problema y los requisitos

Imagina que eres un ingeniero de IA que trabaja para un gran sistema hospitalario en los Estados Unidos. A sus partes interesadas les gustaría una mayor visibilidad de los datos en constante cambio que recopilan. *Quieren respuestas a preguntas ad hoc sobre pacientes, visitas, médicos, hospitales y pagadores de seguros* sin tener que entender un lenguaje de consulta como SQL, solicitar un informe a un analista o esperar a que alguien construya un panel de control.

Para lograr esto, sus partes interesadas quieren una herramienta interna de chatbot, similar a ChatGPT, que pueda responder a preguntas sobre los datos de su empresa. Después de reunirse para reunir los requisitos, se le proporciona una lista de los tipos de preguntas que su chatbot debe responder:

- ¿Cuál es el tiempo de espera actual en el hospital XYZ?
- ¿Qué hospital tiene actualmente el tiempo de espera más corto?
- ¿En qué hospitales se quejan los pacientes de problemas de facturación y seguro?
- ¿Algún paciente se ha quejado de que el hospital está sucio?
- ¿Qué han dicho los pacientes sobre cómo los médicos y las enfermeras se comunican con ellos?
- ¿Qué dicen los pacientes sobre el personal de enfermería del hospital XYZ?
- ¿Cuál fue el importe total de facturación que se cargó a los pagadores de [Cigna](https://en.wikipedia.org/wiki/Cigna) en 2023?
- ¿Cuántos pacientes tiene el Dr. ¿John Doe trató?
- ¿Cuántas visitas están abiertas y cuál es su duración promedio en días?
- ¿Qué médico tiene la duración media de la visita más baja en días?
- ¿Cuánto se facturaro por la estancia del paciente 789?
- ¿Qué hospital trabajó con la mayoría de los pacientes de Cigna en 2023?
- ¿Cuál es el importe medio de facturación de las visitas de emergencia por hospital?
- ¿Qué estado tuvo el mayor porcentaje de aumento de las visitas a los inedicaid de 2022 a 2023?

Puedes responder a preguntas como *¿Cuál fue el monto total de facturación que se cargó a los pagadores de Cigna en 2023?* con estadísticas agregadas utilizando un lenguaje de consulta como SQL. Fundamentalmente, estas preguntas tienen una única respuesta objetiva. Podría ejecutar consultas predefinidas para responderlas, pero cada vez que una parte interesada tenga una pregunta nueva o ligeramente matizada, tiene que escribir una nueva consulta. Para evitar esto, tu chatbot debe generar dinámicamente consultas precisas.

Preguntas como ¿*Algún paciente se ha quejado de que el hospital está sucio?* o *¿Qué han dicho los pacientes sobre cómo los médicos y las enfermeras se comunican con ellos?* son más subjetivos y podrían tener muchas respuestas aceptables. Su chatbot tendrá que leer documentos, como las reseñas de los pacientes, para responder a este tipo de preguntas.

En última instancia, sus partes interesadas quieren una única interfaz de chat que pueda responder sin problemas a preguntas subjetivas y objetivas. Esto significa que, cuando se le presenta una pregunta, su chatbot necesita saber qué tipo de pregunta se está haciendo y de qué fuente de datos extraer.

Por ejemplo, si se le pregunta *¿Cuánto se facturó por la estancia del paciente 789?*, su chatbot debería saber que necesita consultar una base de datos para encontrar la respuesta. Si se le pregunta *¿Qué han dicho los pacientes sobre cómo los médicos y las enfermeras se comunican con ellos?*, su chatbot debería saber que necesita leer y resumir las reseñas de los pacientes.

A continuación, explorará los datos que registra su sistema hospitalario, que es posiblemente el requisito previo más importante para construir su chatbot.

### Explorar los datos disponibles

Antes de construir su chatbot, necesita una comprensión profunda de los datos que utilizará para responder a las consultas de los usuarios. Esto le ayudará a determinar qué es factible y cómo desea estructurar los datos para que su chatbot pueda acceder fácilmente a ellos. Todos los datos que utilizarás en este artículo se generaron sintéticamente, y gran parte de ellos se derivaron de un popular [conjunto de datos de atención médica](https://www.kaggle.com/datasets/prasad22/healthcare-dataset) en Kaggle.

En la práctica, los siguientes conjuntos de datos probablemente se almacenarían como tablas en una base de datos SQL, pero trabajarás con archivos CSV para mantener el enfoque en la construcción del chatbot. Esta sección le dará una descripción detallada de cada archivo CSV.

You’ll need to place all CSV files that are part of this project in your `data/` folder before continuing with the tutorial. Make sure that you downloaded them from the materials and placed them in your `data/` folder:

**Obtenga su código:** [Haga clic aquí para descargar el código fuente gratuito](https://realpython.com/bonus/build-llm-rag-chatbot-with-langchain-code/) de su chatbot LangChain.

#### hospitales.csv

The `hospitals.csv` file records information on each hospital that your company manages. There 30 hospitals and three [fields](https://en.wikipedia.org/wiki/Field_(computer_science)) in this file:

- `hospital_id`: Un número entero que identifica de forma única a un hospital.
- `hospital_name`: El nombre del hospital.
- `hospital_state`: El estado en el que se encuentra el hospital.

Si está familiarizado con las bases de datos SQL tradicionales y el  [star schema](https://en.wikipedia.org/wiki/Star_schema), puede pensar en hospitals.csv como una  [dimension table](https://en.wikipedia.org/wiki/Star_schema#Dimension_tables). Las tablas de dimensiones son relativamente cortas y contienen información descriptiva o atributos que proporcionan contexto a los datos en las  [fact tables](https://en.wikipedia.org/wiki/Star_schema#Fact_tables).. Las tablas de datos registran eventos sobre las entidades almacenadas en tablas de dimensiones, y tienden a ser tablas más largas.

En este caso, `hospitals.csv` registra información específica de los hospitales, pero puede unirse a las tablas de datos para responder a preguntas sobre qué pacientes, médicos y pagadores están relacionados con el hospital. Esto será más claro cuando explores `visits.csv`.

Si tienes curiosidad, puedes inspeccionar las primeras filas de `hospitals.csv` usando una biblioteca de marcos de datos como [Polars](https://realpython.com/polars-python/#the-python-polars-library). Asegúrese de que Polars esté [installed](https://realpython.com/polars-python/#installing-python-polars) en su [virtual environment](https://realpython.com/python-virtual-environments-a-primer/)  y ejecute el siguiente código:

```
>>> import polars as pl

>>> HOSPITAL_DATA_PATH = "data/hospitals.csv"
>>> data_hospitals = pl.read_csv(HOSPITAL_DATA_PATH)

>>> data_hospitals.shape
(30, 3)

>>> data_hospitals.head()
shape: (5, 3)
┌─────────────┬───────────────────────────┬────────────────┐
│ hospital_id ┆ hospital_name             ┆ hospital_state │
│ ---         ┆ ---                       ┆ ---            │
│ i64         ┆ str                       ┆ str            │
╞═════════════╪═══════════════════════════╪════════════════╡
│ 0           ┆ Wallace-Hamilton          ┆ CO             │
│ 1           ┆ Burke, Griffin and Cooper ┆ NC             │
│ 2           ┆ Walton LLC                ┆ FL             │
│ 3           ┆ Garcia Ltd                ┆ NC             │
│ 4           ┆ Jones, Brown and Murray   ┆ NC             │
└─────────────┴───────────────────────────┴────────────────┘
```

En este bloque de código, importa Polars, define la ruta a `hospitals.csv`, lee los datos en un marco de datos Polars, muestra la forma de los datos y muestra las primeras 5 filas. Esto le muestra, por ejemplo, que el hospital **Walton, LLC** tiene una identificación de **2** y se encuentra en el estado de Florida, **FL**.

#### médicos.csv (physicians.csv)

The `physicians.csv` file contains data about the physicians that work for your hospital system. This dataset has the following fields:

- `physician_id`: Un número entero que identifica de forma única a cada médico.
- `physician_name`: El nombre del médico.
- `physician_dob`: La fecha de nacimiento del médico.
- `physician_grad_year`: El año en que el médico se graduó de la escuela de medicina.
- `medical_school`: Donde el médico asistió a la escuela de medicina.
- `salary`: El salario del médico.

Estos datos se pueden volver a considerar como una tabla de dimensiones, y puede inspeccionar las primeras filas usando Polars:

```
PHYSICIAN_DATA_PATH = "data/physicians.csv"
>>> data_physician = pl.read_csv(PHYSICIAN_DATA_PATH)

>>> data_physician.shape
(500, 6)

>>> data_physician.head()
shape: (5, 6)
┌──────────────────┬──────────────┬───────────────┬─────────────────────┬───────────────────────────────────┬───────────────┐
│ physician_name   ┆ physician_id ┆ physician_dob ┆ physician_grad_year ┆ medical_school                    ┆ salary        │
│ ---              ┆ ---          ┆ ---           ┆ ---                 ┆ ---                               ┆ ---           │
│ str              ┆ i64          ┆ str           ┆ str                 ┆ str                               ┆ f64           │
╞══════════════════╪══════════════╪═══════════════╪═════════════════════╪═══════════════════════════════════╪═══════════════╡
│ Joseph Johnson   ┆ 0            ┆ 1970-02-22    ┆ 2000-02-22          ┆ Johns Hopkins University School … ┆ 309534.155076 │
│ Jason Williams   ┆ 1            ┆ 1982-12-22    ┆ 2012-12-22          ┆ Mayo Clinic Alix School of Medic… ┆ 281114.503559 │
│ Jesse Gordon     ┆ 2            ┆ 1959-06-03    ┆ 1989-06-03          ┆ David Geffen School of Medicine … ┆ 305845.584636 │
│ Heather Smith    ┆ 3            ┆ 1965-06-15    ┆ 1995-06-15          ┆ NYU Grossman Medical School       ┆ 295239.766689 │
│ Kayla Hunter DDS ┆ 4            ┆ 1978-10-19    ┆ 2008-10-19          ┆ David Geffen School of Medicine … ┆ 298751.355201 │
└──────────────────┴──────────────┴───────────────┴─────────────────────┴───────────────────────────────────┴───────────────┘
```

Como se puede ver en el bloque de código, hay 500 médicos en `physicians.csv`. Las primeras filas de `physicians.csv` te dan una idea de cómo se ven los datos. Por ejemplo, Heather Smith tiene una identificación médica de 3, nació el 15 de junio de 1965, se graduó de la escuela de medicina el 15 de junio de 1995, asistió a la Escuela de Medicina Grossman de la Universidad de Nueva York, y su salario es de unos 295.239 dólares.

#### pagadores.csv (payers.csv)

El siguiente archivo, `payers.csv`, registra información sobre las compañías de seguros que sus hospitales facturan por las visitas a los pacientes. Similar a `hospitals.csv`, es un archivo pequeño con un par de campos:

- `payer_id`: Un número entero que identifica de forma única a cada pagador.
- `payer_name`: Nombre de la empresa del pagador.

Los únicos cinco pagadores en los datos son **Medicaid, UnitedHealthcare, Aetna, Cigna*** y **Blue Cross***. Sus partes interesadas están muy interesadas en la actividad del pagador, por lo que `payers.csv` será útil una vez que esté conectado con pacientes, hospitales y médicos.

#### reseñas.csv (reviews.csv)

El archivo `reviews.csv` contiene reseñas de pacientes sobre su experiencia en el hospital. Tiene estos campos:

- `review_id`: Un entero que identifica de forma única una revisión.
- `visit_id`: Un número entero que identifica la visita del paciente de la que se trataba la revisión.
- `review`: Esta es la revisión de texto de forma libre dejada por el paciente.
- `physician_name`: El nombre del médico que trató al paciente.
- `hospital_name`: El hospital donde se quedó el paciente.
- `patient_name`: El nombre del paciente.

Este conjunto de datos es el primero que ha visto que contiene el campo de **revisión** de texto libre, y su chatbot debe usarlo para responder preguntas sobre los detalles de la revisión y las experiencias de los pacientes.

Así es como se ve `reviews.csv`:

```
>>> REVIEWS_DATA_PATH = "data/reviews.csv"
>>> data_reviews = pl.read_csv(REVIEWS_DATA_PATH)

>>> data_reviews.shape
(1005, 6)

>>> data_reviews.head()
shape: (5, 6)
┌───────────┬──────────┬───────────────────────────────────┬─────────────────────┬──────────────────┬──────────────────┐
│ review_id ┆ visit_id ┆ review                            ┆ physician_name      ┆ hospital_name    ┆ patient_name     │
│ ---       ┆ ---      ┆ ---                               ┆ ---                 ┆ ---              ┆ ---              │
│ i64       ┆ i64      ┆ str                               ┆ str                 ┆ str              ┆ str              │
╞═══════════╪══════════╪═══════════════════════════════════╪═════════════════════╪══════════════════╪══════════════════╡
│ 0         ┆ 6997     ┆ The medical staff at the hospita… ┆ Laura Brown         ┆ Wallace-Hamilton ┆ Christy Johnson  │
│ 9         ┆ 8138     ┆ The hospital's commitment to pat… ┆ Steven Watson       ┆ Wallace-Hamilton ┆ Anna Frazier     │
│ 11        ┆ 680      ┆ The hospital's commitment to pat… ┆ Chase Mcpherson Jr. ┆ Wallace-Hamilton ┆ Abigail Mitchell │
│ 892       ┆ 9846     ┆ I had a positive experience over… ┆ Jason Martinez      ┆ Wallace-Hamilton ┆ Kimberly Rivas   │
│ 822       ┆ 7397     ┆ The medical team at the hospital… ┆ Chelsey Davis       ┆ Wallace-Hamilton ┆ Catherine Yang   │
└───────────┴──────────┴───────────────────────────────────┴─────────────────────┴──────────────────┴──────────────────┘
```

Hay 1005 reseñas en este conjunto de datos, y puedes ver cómo cada revisión se relaciona con una visita. Por ejemplo, la revisión con ID 9 corresponde a la visita ID 8138, y las primeras palabras son "El compromiso del hospital de acariciar...". Es posible que se pregunte cómo puede conectar una revisión con un paciente, o más generalmente, cómo puede conectar todos los conjuntos de datos descritos hasta ahora entre sí. Aquí es donde entra en juego `visits.csv`.

#### visitas.csv

El último archivo, `visits.csv`, registra detalles sobre cada visita al hospital que su empresa ha atendido. Continuando con la analogía del esquema estelar, puedes pensar en `visits.csv` como una [fact table]([Star schema - Wikipedia](https://en.wikipedia.org/wiki/Star_schema#Fact_tables)) que conecta hospitales, médicos, pacientes y pagadores. Aquí están los campos:

- `visit_id`: El identificador único de una visita al hospital.
- `patient_id`: La identificación del paciente asociada con la visita.
- `date_of_admission`: La fecha en que el paciente fue admitido en el hospital.
- `room_number`: El número de habitación del paciente.
- `admission_type`: Uno de "Electivo", "Emergencia" o "Urgente".
- `chief_complaint`: Una cadena que describe la razón principal del paciente para estar en el hospital.
- `primary_diagnosis`: Una cadena que describe el diagnóstico primario realizado por el médico.
- `treatment_description`: Un resumen textual del tratamiento dado por el médico.
- `test_results`: Uno de "Inconclusa", "Normal" o "Anormal".
- `discharge_date`: La fecha en que el paciente fue dado de alta del hospital
- `physician_id`: La identificación del médico que trató al paciente.
- `hospital_id`: La identificación del hospital en el que se quedó el paciente.
- `payer_id`: La identificación del pagador del seguro utilizada por el paciente.
- `billing_amount`: La cantidad de dinero facturada al pagador por la visita.
- `visit_status`: Uno de "ABIERTO" o "DESCARGADO".

Este conjunto de datos le da todo lo que necesita para responder preguntas sobre la relación entre cada entidad hospitalaria. Por ejemplo, si conoce una identificación de médico, puede usar `visits.csv` para averiguar con qué pacientes, pagadores y hospitales está asociado el médico. Echa un vistazo a cómo se ve `visits.csv` en Polars:

```
>>> VISITS_DATA_PATH = "data/visits.csv"
>>> data_visits = pl.read_csv(VISITS_DATA_PATH)

>>> data_visits.shape
(9998, 15)

>>> data_visits.head()
shape: (5, 15)
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ patient ┆ date_of ┆ billing ┆ room_nu ┆ admissi ┆ dischar ┆ test_r ┆ visit_ ┆ physic ┆ payer_ ┆ hospit ┆ chief_ ┆ treatm ┆ primar ┆ visit_ │
│ _id     ┆ _admiss ┆ _amount ┆ mber    ┆ on_type ┆ ge_date ┆ esults ┆ id     ┆ ian_id ┆ id     ┆ al_id  ┆ compla ┆ ent_de ┆ y_diag ┆ status │
│ ---     ┆ ion     ┆ ---     ┆ ---     ┆ ---     ┆ ---     ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ int    ┆ script ┆ nosis  ┆ ---    │
│ i64     ┆ ---     ┆ f64     ┆ i64     ┆ str     ┆ str     ┆ str    ┆ i64    ┆ i64    ┆ i64    ┆ i64    ┆ ---    ┆ ion    ┆ ---    ┆ str    │
│         ┆ str     ┆         ┆         ┆         ┆         ┆        ┆        ┆        ┆        ┆        ┆ str    ┆ ---    ┆ str    ┆        │
│         ┆         ┆         ┆         ┆         ┆         ┆        ┆        ┆        ┆        ┆        ┆        ┆ str    ┆        ┆        │
╞═════════╪═════════╪═════════╪═════════╪═════════╪═════════╪════════╪════════╪════════╪════════╪════════╪════════╪════════╪════════╪════════╡
│ 0       ┆ 2022-11 ┆ 37490.9 ┆ 146     ┆ Electiv ┆ 2022-12 ┆ Inconc ┆ 0      ┆ 102    ┆ 1      ┆ 0      ┆ null   ┆ null   ┆ null   ┆ DISCHA │
│         ┆ -17     ┆ 83364   ┆         ┆ e       ┆ -01     ┆ lusive ┆        ┆        ┆        ┆        ┆        ┆        ┆        ┆ RGED   │
│ 1       ┆ 2023-06 ┆ 47304.0 ┆ 404     ┆ Emergen ┆ null    ┆ Normal ┆ 1      ┆ 435    ┆ 4      ┆ 5      ┆ null   ┆ null   ┆ null   ┆ OPEN   │
│         ┆ -01     ┆ 64845   ┆         ┆ cy      ┆         ┆        ┆        ┆        ┆        ┆        ┆        ┆        ┆        ┆        │
│ 2       ┆ 2019-01 ┆ 36874.8 ┆ 292     ┆ Emergen ┆ 2019-02 ┆ Normal ┆ 2      ┆ 348    ┆ 2      ┆ 6      ┆ null   ┆ null   ┆ null   ┆ DISCHA │
│         ┆ -09     ┆ 96997   ┆         ┆ cy      ┆ -08     ┆        ┆        ┆        ┆        ┆        ┆        ┆        ┆        ┆ RGED   │
│ 3       ┆ 2020-05 ┆ 23303.3 ┆ 480     ┆ Urgent  ┆ 2020-05 ┆ Abnorm ┆ 3      ┆ 270    ┆ 4      ┆ 15     ┆ null   ┆ null   ┆ null   ┆ DISCHA │
│         ┆ -02     ┆ 22092   ┆         ┆         ┆ -03     ┆ al     ┆        ┆        ┆        ┆        ┆        ┆        ┆        ┆ RGED   │
│ 4       ┆ 2021-07 ┆ 18086.3 ┆ 477     ┆ Urgent  ┆ 2021-08 ┆ Normal ┆ 4      ┆ 106    ┆ 2      ┆ 29     ┆ Persis ┆ Prescr ┆ J45.90 ┆ DISCHA │
│         ┆ -09     ┆ 44184   ┆         ┆         ┆ -02     ┆        ┆        ┆        ┆        ┆        ┆ tent   ┆ ibed a ┆ 9 -    ┆ RGED   │
│         ┆         ┆         ┆         ┆         ┆         ┆        ┆        ┆        ┆        ┆        ┆ cough  ┆ combin ┆ Unspec ┆        │
│         ┆         ┆         ┆         ┆         ┆         ┆        ┆        ┆        ┆        ┆        ┆ and    ┆ ation  ┆ ified  ┆        │
│         ┆         ┆         ┆         ┆         ┆         ┆        ┆        ┆        ┆        ┆        ┆ shortn ┆ of     ┆ asthma ┆        │
│         ┆         ┆         ┆         ┆         ┆         ┆        ┆        ┆        ┆        ┆        ┆ ess o… ┆ inha…  ┆ , un…  ┆        │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

You can see there are **9998** visits recorded along with the 15 fields described above. Notice that `chief_complaint`, `treatment_description`, and `primary_diagnosis` might be missing for a visit. You’ll have to keep this in mind as your stakeholders might not be aware that many visits are missing critical data—this may be a valuable insight in itself! Lastly, notice that when a visit is still open, the `discharged_date` will be missing.

Puedes ver que hay **9998*** visitas registradas junto con los 15 campos descritos anteriormente. Tenga en cuenta que `chief_complaint`, `treatment_description` y `primary_diagnosis` podrían faltar para una visita. Tendrá que tener esto en cuenta, ya que es posible que sus partes interesadas no sean conscientes de que a muchas visitas les faltan datos críticos, ¡esto puede ser una visión valiosa en sí mismo! Por último, tenga en cuenta que cuando una visita todavía está abierta, `discharged_date` se perdera.

Ahora tiene una comprensión de los datos que utilizará para construir el chatbot que sus partes interesadas desean. En resumen, los archivos se desglosan para simular cómo podría ser una base de datos SQL tradicional. Cada hospital, paciente, médico, revisión y pagador están conectados a través de `visits.csv`.

#### Tiempos de espera

Es posible que hayas notado que no hay datos para responder a preguntas como *¿Cuál es el tiempo de espera actual en el hospital XYZ?* *¿*O *qué hospital tiene actualmente el tiempo de espera más corto?* Desafortunadamente, el sistema hospitalario no registra los tiempos de espera históricos. Su chatbot tendrá que llamar a una API para obtener información actual sobre el tiempo de espera. Verás cómo funciona esto más adelante.

Con una comprensión de los requisitos del negocio, los datos disponibles y las funcionalidades de LangChain, puede crear un diseño para su chatbot.

### Diseña el Chatbot

Ahora que conoce los requisitos comerciales, los datos y los requisitos previos de LangChain, está listo para diseñar su chatbot. Un buen diseño le da a usted y a otros una comprensión conceptual de los componentes necesarios para construir su chatbot. Su diseño debe ilustrar claramente cómo fluyen los datos a través de su chatbot, y debe servir como una referencia útil durante el desarrollo.

Su chatbot utilizará múltiples herramientas para responder a diversas preguntas sobre su sistema hospitalario. Aquí hay un diagrama de flujo que ilustra cómo lograrás esto:

[![Diagrama de flujo del chatbot](https://files.realpython.com/media/Screenshot_2024-01-15_at_8.08.18_PM.fe16f8a318cc.png)](https://files.realpython.com/media/Screenshot_2024-01-15_at_8.08.18_PM.fe16f8a318cc.png)

Arquitectura y flujo de datos para el chatbot del sistema hospitalario

Este diagrama de flujo ilustra cómo se mueven los datos a través de su chatbot, desde la consulta de entrada del usuario hasta la respuesta final. Aquí hay un resumen de cada componente:

- **Agente de LangChain**: El agente de LangChain es el cerebro de tu chatbot. Dada una consulta del usuario, el agente decide a qué herramienta llamar y qué dar a la herramienta como entrada. A continuación, el agente observa el resultado de la herramienta y decide qué devolver al usuario; esta es la respuesta del agente.
- **Neo4j AuraDB**: almacenará tanto los datos estructurados del sistema hospitalario como las revisiones de los pacientes en una base de datos de gráficos Neo4j AuraDB. Aprenderás todo sobre esto en la siguiente sección.
- **Cadena de cifrado LangChain Neo4j**: Esta cadena intenta convertir la consulta del usuario en Cypher, el lenguaje de consulta de Neo4j, y ejecutar la consulta de Cypher en Neo4j. A continuación, la cadena responde a la consulta del usuario utilizando los resultados de la consulta de Cypher. La respuesta de la cadena se devuelve al agente de LangChain y se envía al usuario.
- **LangChain Neo4j revisa la cadena vectorial**: Esto es muy similar a la cadena que construyó en el paso 1, excepto que ahora las incrustaciones de revisión de pacientes se almacenan en Neo4j. La cadena busca reseñas relevantes basadas en aquellas semánticamente similares a la consulta del usuario, y las revisiones se utilizan para responder a la consulta del usuario.
- **Función de tiempos de espera**: similar a la lógica del paso 1, el agente de LangChain intenta extraer un nombre de hospital de la consulta del usuario. El nombre del hospital se pasa como entrada a una función de Python que obtiene tiempos de espera, y el tiempo de espera se devuelve al agente.

Para ver un ejemplo, supongamos que un usuario pregunta *¿Cuántas visitas de emergencia hubo en 2023?* El agente de LangChain recibirá esta pregunta y decidirá a qué herramienta, si la hay, pasar la pregunta. En este caso, el agente debe pasar la pregunta a la *cadena de cifrado LangChain Neo4j*. La cadena intentará convertir la pregunta en una consulta de cifrado, ejecutar la consulta de cifrado en Neo4j y utilizar los resultados de la consulta para responder a la pregunta.

Una vez que la cadena de cifrado LangChain Neo4j responda a la pregunta, devolverá la respuesta al agente, y el agente transmitirá la respuesta al usuario.

Con este diseño en mente, puedes empezar a construir tu chatbot. Tu primera tarea es configurar una instancia de Neo4j AuraDB para que tu chatbot acceda.

## Paso 3: Configurar una base de datos de gráficos Neo4j

Como vio en el paso 2, los datos de su sistema hospitalario se almacenan actualmente en archivos CSV. Antes de crear su chatbot, debe almacenar estos datos en una base de datos que su chatbot pueda consultar. Utilizarás [Neo4j AuraDB](https://neo4j.com/cloud/aura-free/) para esto.

Antes de aprender a configurar una instancia de Neo4j AuraDB, obtendrá una visión general de las bases de datos de gráficos, y verá por qué el uso de una base de datos de gráficos puede ser una mejor opción que una base de datos relacional para este proyecto.

### Una breve descripción de las bases de datos de gráficos

Las bases de datos de gráficos, como Neo4j, son bases de datos diseñadas para representar y procesar los datos almacenados como un gráfico. Los datos del gráfico consisten en **nodos**, **bordes** o **relaciones**, y **propiedades**. Los nodos representan entidades, las relaciones conectan a las entidades y las propiedades proporcionan metadatos adicionales sobre los nodos y las relaciones.

Por ejemplo, así es como podrías representar los nodos y las relaciones del sistema hospitalario en un gráfico:

[![Ejemplo de datos gráficos](https://files.realpython.com/media/Screenshot_2024-01-16_at_4.33.31_PM.043fc98132e3.png)](https://files.realpython.com/media/Screenshot_2024-01-16_at_4.33.31_PM.043fc98132e3.png)

Gráfico del sistema hospitalario

Este gráfico tiene tres nodos: **Paciente**, **Visita** y **Pagador**. **El paciente** y **la visita** están conectados por la relación **HAS**, lo que indica que un paciente del hospital tiene una visita. Del mismo modo, **la visita** y **el pagador** están conectados por la relación **COVERED_BY**, lo que indica que un pagador de seguro cubre una visita al hospital.

Observe cómo las relaciones están representadas por una flecha que indica su dirección. Por ejemplo, la dirección de la relación **HAS** le dice que un paciente puede tener una visita, pero una visita no puede tener un paciente.

Tanto los nodos como las relaciones pueden tener propiedades. En este ejemplo, los nodos **del paciente** tienen propiedades de identificación, nombre y fecha de nacimiento, y la relación **COVERED_BY** tiene propiedades de fecha de servicio y cantidad de facturación. Almacenar datos en un gráfico como este tiene varias ventajas:

1. **Simplicidad**: Modelar las relaciones del mundo real entre entidades es natural en las bases de datos de gráficos, lo que reduce la necesidad de esquemas complejos que requieren múltiples operaciones de unión para responder a las consultas.

2. **Relaciones**: Las bases de datos de gráficos sobresalen en el manejo de relaciones complejas. Atravesar las relaciones es eficiente, lo que facilita la consulta y el análisis de los datos conectados.

3. **Flexibilidad**: las bases de datos de gráficos no tienen esquemas, lo que permite una fácil adaptación a las estructuras de datos cambiantes. Esta flexibilidad es beneficiosa para la evolución de los modelos de datos.

4. **Rendimiento**: La recuperación de datos conectados es más rápida en las bases de datos de gráficos que en las bases de datos relacionales, especialmente para escenarios que involucran consultas complejas con múltiples relaciones.

5. **Coincidencia de patrones**: las bases de datos de gráficos admiten potentes consultas de coincidencia de patrones, lo que facilita la expresión y la búsqueda de estructuras específicas dentro de los datos.

Cuando tienes datos con muchas relaciones complejas, la simplicidad y flexibilidad de las bases de datos de gráficos las hace más fáciles de diseñar y consultar en comparación con las bases de datos relacionales. Como verás más adelante, especificar relaciones en las consultas de la base de datos de gráficos es concisa y no implica uniones complicadas. Si estás interesado, Neo4j lo ilustra bien con una base de datos de ejemplos realista en su [documentación](https://neo4j.com/developer/cypher/guide-sql-to-cypher/).

Debido a esta representación concisa de datos, hay menos margen de error cuando un LLM genera consultas a la base de datos de gráficos. Esto se debe a que solo necesita decirle a LLM sobre los nodos, las relaciones y las propiedades de su base de datos de gráficos. Compare esto con las bases de datos relacionales donde el LLM debe navegar y retener el conocimiento de los esquemas de la tabla y las relaciones de clave externa en toda su base de datos, dejando más espacio para errores en la generación de SQL.

A continuación, comenzarás a trabajar con bases de datos de gráficos configurando una instancia de [Neo4j AuraDB](https://neo4j.com/cloud/aura-free/). Después de eso, moverás el sistema hospitalario a tu instancia de Neo4j y aprenderás a consultarlo.

### Crear una cuenta Neo4j y una instancia de AuraDB

Para empezar a usar Neo4j, puedes crear una cuenta gratuita de [Neo4j AuraDB](https://neo4j.com/cloud/aura-free/). La página de destino debería tener un aspecto similar a esto:

[![Pantalla de inicio de Neo4j Aura](https://files.realpython.com/media/Screenshot_2024-01-10_at_9.52.26_AM.13dfb78c613b.png)](https://files.realpython.com/media/Screenshot_2024-01-10_at_9.52.26_AM.13dfb78c613b.png)

Pantalla de inicio Neo4j Aura

Haga clic en el botón **Iniciar gratis** y cree una cuenta. Una vez que hayas iniciado sesión, deberías ver la consola Neo4j Aura:

Haga clic en el botón **Iniciar gratis** y cree una cuenta. Una vez que hayas iniciado sesión, deberías ver la consola Neo4j Aura:

[![Consola Neo4j Aura](https://files.realpython.com/media/Screenshot_2024-01-10_at_9.53.58_AM.c9d5252982fc.png)](https://files.realpython.com/media/Screenshot_2024-01-10_at_9.53.58_AM.c9d5252982fc.png)

Crear una nueva instancia de Aura

Haga clic en **Nueva instancia** y cree una instancia gratuita. Debería aparecer un modal similar a este:

[![Neo4j Aura Crear Una Nueva Instancia](https://files.realpython.com/media/Screenshot_2024-01-10_at_9.56.24_AM.6d833ddf5733.png)](https://files.realpython.com/media/Screenshot_2024-01-10_at_9.56.24_AM.6d833ddf5733.png)

Nuevo modal de instancia de Aura

Después de hacer clic en **Descargar y continuar**, se debe crear su instancia y se debe descargar un archivo de texto que contiene las credenciales de la base de datos Neo4j. Una vez creada la instancia, verás que su estado está en **ejecución**. Todavía no debería haber nodos ni relaciones:

[![Instancia en ejecución Neo4j Aura](https://files.realpython.com/media/Screenshot_2024-01-10_at_10.00.34_AM.0ca76879f1fc.png)](https://files.realpython.com/media/Screenshot_2024-01-10_at_10.00.34_AM.0ca76879f1fc.png)

Instancia en ejecución de Aura

A continuación, abra el archivo de texto que descargó con sus credenciales de Neo4j y copie `NEO4J_URI`, `NEO4J_USERNAME` y `NEO4J_PASSWORD` en su archivo `.env`:

```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>

NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_URI>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>
```

Utilizará estas variables de entorno para conectarse a su instancia de Neo4j en Python para que su chatbot pueda ejecutar consultas.

> **Nota**: De forma predeterminada, su **NEO4J_URI** debe ser similar a **neo4j+s://.databases.neo4j.io**. El esquema de URL **neo4j+s** solo utiliza [certificados firmados por CA](https://en.wikipedia.org/wiki/Certificate_authority), que podrían no funcionar para usted. Si este es el caso, cambie su URI para usar el esquema de URL theneo4j**+ssc** - **neo4j+ssc://.databases.neo4j.io**. Puedes leer más sobre lo que esto significa en la documentación de Neo4j sobre [protocolos de conexión y seguridad](https://neo4j.com/docs/python-manual/current/connect-advanced/#_connection_protocols_and_security).

Ahora tienes todo en su lugar para interactuar con tu instancia de Neo4j. A continuación, diseñarás la base de datos de gráficos del sistema hospitalario. Esto le dirá cómo están relacionadas las entidades del hospital, e informará el tipo de consultas que puede ejecutar.

### Diseñar la base de datos de gráficos del sistema hospitalario

Ahora que tiene una instancia de Neo4j AuraDB en ejecución, necesita decidir qué nodos, relaciones y propiedades desea almacenar. Una de las formas más populares de representar esto es con un diagrama de flujo. Basándose en su comprensión de los datos del sistema hospitalario, se le ocurre el siguiente diseño:

[![Ontología del sistema hospitalario](https://files.realpython.com/media/Screenshot_2024-01-11_at_9.25.30_AM.16896d00ee08.png)](https://files.realpython.com/media/Screenshot_2024-01-11_at_9.25.30_AM.16896d00ee08.png)

Diseño de la base de datos de gráficos del sistema hospitalario

Este diagrama muestra todos los nodos y relaciones en los datos del sistema hospitalario. Una forma útil de pensar en este diagrama de flujo es comenzar con el nodo **del paciente** y seguir las relaciones. Un **paciente** **tiene** una **visita** **en** un **hospital**, y el **hospital** **emplea a** un **médico** para **tratar** la **visita**, que está **cubierta por** un **pagador** del seguro.

Estas son las propiedades almacenadas en cada nodo:

[![Propiedades del nodo gráfico del hospital](https://files.realpython.com/media/Screenshot_2024-01-17_at_8.28.33_AM.e784ec79aa41.png)](https://files.realpython.com/media/Screenshot_2024-01-17_at_8.28.33_AM.e784ec79aa41.png)

Propiedades del nodo del sistema hospitalario

La mayoría de estas propiedades provienen directamente de los campos que exploraste en el paso 2. Una diferencia notable es que los nodos de **revisión** tienen una propiedad **de incrustación**, que es una representación vectorial de las propiedades de **nombre_paciente**, **nombre_**de**_médico** y **texto**. Esto le permite hacer [búsquedas vectoriales](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/) sobre nodos de revisión como lo hizo con ChromaDB.

Estas son las propiedades de la relación:

[![Hospital Graph Relationship Properties](https://files.realpython.com/media/Screenshot_2024-01-17_at_9.07.16_AM.de07d986e379.png)](https://files.realpython.com/media/Screenshot_2024-01-17_at_9.07.16_AM.de07d986e379.png)

Propiedades de la relación del sistema hospitalario

Como puede ver, **COVERED_BY** es la única relación con más de una propiedad **de identificación**. La **fecha_de_servicio** es la fecha en que el paciente fue dado de alta de una visita, y **la cantidad_de facturación** es la cantidad que se cobra al pagador por la visita.

> **Nota**: Estos datos falsos del sistema hospitalario tienen un número relativamente pequeño de nodos y relaciones de lo que normalmente se vería en un entorno empresarial. Sin embargo, puedes imaginar fácilmente cuántos nodos y relaciones más podrías añadir para un sistema hospitalario real. Por ejemplo, las enfermeras, los farmacéuticos, las farmacias, los medicamentos recetados, las cirugías, los familiares de los pacientes y muchas más entidades hospitalarias podrían ser representados como nodos.
> 
> También podría rediseñar esto para que los diagnósticos y los síntomas se representen como nodos en lugar de propiedades, o podría agregar más propiedades de relación. Podrías hacer todo esto sin cambiar el diseño que ya tienes. Esta es la belleza de los gráficos: simplemente agregas más nodos y relaciones a medida que tus datos evolucionan.

Ahora que tienes una visión general del diseño del sistema hospitalario que usarás, ¡es hora de mover tus datos a Neo4j!

### Subir datos a Neo4j

Con una instancia Neo4j en ejecución y una comprensión de los nodos, propiedades y relaciones que desea almacenar, puede mover los datos del sistema hospitalario a Neo4j. Para ello, crearás una carpeta llamada `hospital_neo4j_etl` con algunos archivos vacíos. También querrás crear un archivo `docker-compose.yml` en el directorio raíz de tu proyecto:

```
./
│
├── hospital_neo4j_etl/
│   │
│   ├── src/
│   │   ├── entrypoint.sh
│   │   └── hospital_bulk_csv_write.py
│   │
│   ├── Dockerfile
│   └── pyproject.toml
│
├── .env
└── docker-compose.yml
```

Su archivo `.env` debe tener las siguientes variables de entorno:

```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>

NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_URI>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>

HOSPITALS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/hospitals.csv
PAYERS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/payers.csv
PHYSICIANS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/physicians.csv
PATIENTS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/patients.csv
VISITS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/visits.csv
REVIEWS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/reviews.csv
```

Ten en cuenta que has almacenado todos los archivos CSV en una ubicación pública en [GitHub](https://github.com/hfhoffman1144/langchain_neo4j_rag_app/tree/main/data). Debido a que su instancia Neo4j AuraDB se está ejecutando en la nube, no puede acceder a los archivos de su máquina local, y tiene que usar HTTP o cargar los archivos directamente en su instancia. Para este ejemplo, puede utilizar el enlace de arriba o cargar los datos en otra ubicación.

**Nota:** Si está cargando datos propietarios a Neo4j, asegúrese siempre de que se almacenen en un lugar seguro y se transfieran adecuadamente. Los datos utilizados para este proyecto son todos sintéticos y no propietarios, por lo que no hay problema en subirlos a través de una conexión HTTP pública. Sin embargo, esto no sería una buena idea en la práctica. Puedes leer más sobre [formas seguras de importar datos a Neo4j](https://neo4j.com/docs/aura/aurads/importing-data/) en su documentación.

Una vez que haya rellenado su archivo `.env`, abra `pyproject.toml`, que proporciona configuración, metadatos y dependencias definidas en el formato [TOML](https://realpython.com/python-toml/):

```toml
TOML hospital_neo4j_etl/pyproject.toml
[project]
name = "hospital_neo4j_etl"
version = "0.1"
dependencies = [
   "neo4j==5.14.1",
   "retry==0.9.2"
]

[project.optional-dependencies]
dev = ["black", "flake8"]
```

Este proyecto es un [proceso de extracción de](https://en.wikipedia.org/wiki/Extract,_transform,_load) huesos desnudos[, transformación, carga (ETL)](https://en.wikipedia.org/wiki/Extract,_transform,_load) que mueve los datos a Neo4j, por lo que sus únicas dependencias son [neo4j](https://pypi.org/project/neo4j/) y [volver a intentarlo](https://pypi.org/project/retry/). El script principal para el ETL eshospital`hospital_neo4j_etl/src/hospital_bulk_csv_write.py`. Es demasiado largo para incluir el script completo aquí, pero tendrás una idea de los pasos principales que se ejecutan `hospital_neo4j_etl/src/hospital_bulk_csv_write.py`. Puedes copiar el guión completo de los materiales:

**Obtenga su código:** [Haga clic aquí para descargar el código fuente gratuito](https://realpython.com/bonus/build-llm-rag-chatbot-with-langchain-code/) de su chatbot LangChain.

En primer lugar, importa dependencias, carga variables de entorno y configura [el registro](https://realpython.com/python-logging/):

Python `hospital_neo4j_etl/src/hospital_bulk_csv_write.py`

```python
import os
import logging
from retry import retry
from neo4j import GraphDatabase

HOSPITALS_CSV_PATH = os.getenv("HOSPITALS_CSV_PATH")
PAYERS_CSV_PATH = os.getenv("PAYERS_CSV_PATH")
PHYSICIANS_CSV_PATH = os.getenv("PHYSICIANS_CSV_PATH")
PATIENTS_CSV_PATH = os.getenv("PATIENTS_CSV_PATH")
VISITS_CSV_PATH = os.getenv("VISITS_CSV_PATH")
REVIEWS_CSV_PATH = os.getenv("REVIEWS_CSV_PATH")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)

# ...
```

Usted importa la clase `GraphDatabase` de `neo4j` para conectarse a su instancia en ejecución. Tenga en cuenta aquí que ya no está usando Python-dotenv para cargar variables de entorno. En su lugar, pasará variables de entorno al contenedor Docker que ejecuta su script. A continuación, definirá las funciones para mover los datos del hospital a Neo4j siguiendo su diseño:

Python `hospital_neo4j_etl/src/hospital_bulk_csv_write.py`

```python
# ...

NODES = ["Hospital", "Payer", "Physician", "Patient", "Visit", "Review"]

def _set_uniqueness_constraints(tx, node):
    query = f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node})
        REQUIRE n.id IS UNIQUE;"""
    _ = tx.run(query, {})


@retry(tries=100, delay=10)
def load_hospital_graph_from_csv() -> None:
    """Load structured hospital CSV data following
    a specific ontology into Neo4j"""

    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    LOGGER.info("Setting uniqueness constraints on nodes")
    with driver.session(database="neo4j") as session:
        for node in NODES:
            session.execute_write(_set_uniqueness_constraints, node)
    # ...

# ...
```

En primer lugar, define una función de ayuda, `_set_uniqueness_constraints()`, que crea y ejecuta consultas que obligan a cada nodo a tener un ID único. En `load_hospital_graph_from_csv()`, crea una instancia de un controlador que se conecta a su instancia Neo4j y establece restricciones de singularidad para cada nodo del sistema hospitalario.

Fíjate en el [decorator]([Primer on Python Decorators – Real Python](https://realpython.com/primer-on-python-decorators/))  `@retry` adjunto a `load_hospital_graph_from_csv()`. Si `load_hospital_graph_from_csv()` falla por cualquier motivo, este decorador lo volverá a ejecutar cien veces con un retraso de diez segundos entre intentos. Esto es útil cuando hay problemas de conexión intermitente con Neo4j que generalmente se resuelven recreando una conexión. Sin embargo, asegúrese de revisar los registros de secuencias de comandos para ver si un error vuelve a ocurrir más de unas cuantas veces.

A continuación, `load_hospital_graph_from_csv()` carga datos para cada nodo y relación:

Python `hospital_neo4j_etl/src/hospital_bulk_csv_write.py`

```python
# ...

@retry(tries=100, delay=10)
def load_hospital_graph_from_csv() -> None:
    """Load structured hospital CSV data following
    a specific ontology into Neo4j"""

    # ...

    LOGGER.info("Loading hospital nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{HOSPITALS_CSV_PATH}' AS hospitals
        MERGE (h:Hospital {{id: toInteger(hospitals.hospital_id),
                            name: hospitals.hospital_name,
                            state_name: hospitals.hospital_state}});
        """
        _ = session.run(query, {})

   # ...

if __name__ == "__main__":
    load_hospital_graph_from_csv()
```

Cada nodo y relación se carga desde sus respectivos archivos csv y se escribe en Neo4j de acuerdo con el diseño de su base de datos de gráficos. Al final del script, llamas a `load_hospital_graph_from_csv()` en el  [name-main idiom](https://realpython.com/if-name-main-python/), y todos los datos deberían rellenarse en tu instancia de Neo4j.

Después de escribir `hospital_neo4j_etl/src/hospital_bulk_csv_write.py`, puede definir un archivo `entrypoint.sh` que se ejecutará cuando se inicie su contenedor Docker:

Shell `hospital_neo4j_etl/src/entrypoint.sh`

```bash
#!/bin/bash

# Run any setup steps or pre-processing tasks here
echo "Running ETL to move hospital data from csvs to Neo4j..."

# Run the ETL script
python hospital_bulk_csv_write.py
```

Este archivo de punto de entrada no es técnicamente necesario para este proyecto, pero es una buena práctica al construir contenedores porque le permite ejecutar los comandos de shell necesarios antes de ejecutar su script principal.

El último archivo que se escribe para su ETL es el archivo Docker. Se ve así:

Dockerfile `hospital_neo4j_etl/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY ./src/ /app

COPY ./pyproject.toml /code/pyproject.toml
RUN pip install /code/.

CMD ["sh", "entrypoint.sh"]
```

Este `Dockerfile` le dice a su contenedor que use la [distribution](https://hub.docker.com/_/python) `python:3.11-slim`, copie el contenido de `hospital_neo4j_etl/src/` en el directorio `/app` dentro del contenedor, instale las dependencias de `pyproject.toml` y ejecute `entrypoint.sh`.

Ahora puedes añadir este proyecto a `docker-compose.yml`:

YAML `docker-compose.yml`

```yaml
version: '3'

services:
  hospital_neo4j_etl:
    build:
      context: ./hospital_neo4j_etl
    env_file:
      - .env
```

El ETL se ejecutará como un servicio llamado `hospital_neo4j_etl`, y ejecutará el Dockerfile en `./hospital_neo4j_etl` utilizando variables de entorno de `.env`. Dado que solo tienes un contenedor, todavía no necesitas un docker-compose. Sin embargo, agregará más contenedores para orquestar con su ETL en la siguiente sección, por lo que es útil comenzar con `docker-compose.yml`.

Para ejecutar su ETL, abra un terminal y ejecute:

Shell

```shell
$ docker-compose up --build
```

Una vez que el ETL termine de ejecutarse, vuelva a su consola Aura:

[![Instancia en ejecución Neo4j Aura](https://files.realpython.com/media/Screenshot_2024-01-10_at_10.00.34_AM.0ca76879f1fc.png)](https://files.realpython.com/media/Screenshot_2024-01-10_at_10.00.34_AM.0ca76879f1fc.png)

Consola Aura

Haga clic en **Abrir** y se le pedirá que introduzca su contraseña de Neo4j. Después de iniciar sesión con éxito en la instancia, debería ver una pantalla similar a esta:

[![Instancia Neo4j Aura con datos hospitalarios cargados](https://files.realpython.com/media/Screenshot_2024-01-12_at_8.14.38_AM.72233e36a1e0.png)](https://files.realpython.com/media/Screenshot_2024-01-12_at_8.14.38_AM.72233e36a1e0.png)

Instancia Neo4j Aura con datos del sistema hospitalario cargados

Como puede ver en **Información de la base de datos**, se cargaron todos los nodos, relaciones y propiedades. Hay 21 187 nodos y 48 259 relaciones. ¡Estás listo para empezar a escribir consultas!

### Consulta el gráfico del sistema hospitalario

Lo último que tienes que hacer antes de crear tu chatbot es familiarizarte con la sintaxis [de Cypher](https://neo4j.com/docs/getting-started/cypher-intro/). Cypher es el lenguaje de consulta de Neo4j, y es bastante intuitivo de aprender, especialmente si estás familiarizado con SQL. Esta sección cubrirá lo básico, y eso es todo lo que necesitas para construir el chatbot. Puede consultar la [documentación de Neo4j](https://neo4j.com/docs/getting-started/cypher-intro/) para obtener una visión general más completa de Cypher.

La palabra clave más utilizada para leer datos en Cypher es `MATCH`, y se utiliza para especificar patrones a buscar en el gráfico. El patrón más simple es uno con un solo nodo. Por ejemplo, si desea encontrar los primeros cinco nodos de pacientes escritos en el gráfico, podría ejecutar la siguiente consulta de cifrado:

Lenguaje de consulta de cifrado

```
MATCH (p:Patient)
RETURN p LIMIT 5;
```

En esta consulta, estás haciendo coincidir en los nodos del `paciente`. En Cypher, los nodos siempre se indican con paréntesis. El `p` en `(p:Paciente)` es un alias al que puede hacer referencia más adelante en la consulta. `RETURN p LIMIT 5;` le dice a Neo4j que solo devuelva cinco nodos de paciente. Puedes ejecutar esta consulta en la interfaz de usuario de Neo4j, y los resultados deberían verse así:

[![Consulta de nodo de coincidencia de cifrado](https://files.realpython.com/media/Screenshot_2024-01-12_at_8.43.18_AM.da207917fbbd.png)](https://files.realpython.com/media/Screenshot_2024-01-12_at_8.43.18_AM.da207917fbbd.png)

Consulta de nodo de coincidencia de cifrado en la interfaz de usuario de Neo4j

La vista **de tabla** le muestra los cinco nodos de **pacientes** devueltos junto con sus propiedades. También puedes explorar el gráfico y la vista en bruto si estás interesado.

Si bien la coincidencia en un solo nodo es sencilla, a veces eso es todo lo que necesitas para obtener información útil. Por ejemplo, si su parte interesada dijo que **me diera un resumen de la visita 56**, la siguiente consulta le da la respuesta:

Lenguaje de consulta de cifrado

```
MATCH (v:Visit)
WHERE v.id = 56
RETURN v;
```

Esta consulta coincide con los nodos de `Visit` que tienen una identificación de 56, especificada por  `WHERE v.id = 56`. Puede filtrar por nodo arbitrario y propiedades de relación en las cláusulas `WHERE`. Los resultados de esta consulta se ven así:

[![Consulta de nodo de coincidencia de cifrado con filtro](https://files.realpython.com/media/Screenshot_2024-01-12_at_9.23.59_AM.cf28876d7d65.png)](https://files.realpython.com/media/Screenshot_2024-01-12_at_9.23.59_AM.cf28876d7d65.png)

Cifrar coincide con la consulta de nodo filtrada en una propiedad de nodo

Desde la salida de la consulta, se puede ver que la **visita** devuelta tiene el **id** 56. Luego podría mirar todas las propiedades de la visita para llegar a un resumen verbal de la visita; esto es lo que hará su cadena de Cypher.

La coincidencia en los nodos es genial, pero el verdadero poder de Cypher proviene de su capacidad para coincidir en los patrones de relación. Esto te da una idea de las relaciones sofisticadas, explotando el poder de las bases de datos de gráficos. Continuando con la consulta **de visita**, probablemente quieras saber a qué **paciente** pertenece la **visita**. Puedes obtener esto de la relación **HAS**:

Lenguaje de consulta de cifrado

```
MATCH (p:Patient)-[h:HAS]->(v:Visit)
WHERE v.id = 56
RETURN v,h,p;
```

Esta consulta de cifrado busca al `Patient` que tiene una `Visit` con `id` 56. Notarás que la relación `HAS` está rodeada de corchetes en lugar de paréntesis, y su direccionalidad está indicada por una flecha. Si intentaste `MATCH` `(p:Patient)<-[h:HAS]-(v:Visit)`, la consulta no devolvería nada porque la dirección de la relación `HAS` es incorrecta.

Los resultados de la consulta se ven así:

[![Consulta de relación de coincidencia de cifrado](https://files.realpython.com/media/Screenshot_2024-01-12_at_9.49.31_AM.3cad959aa115.png)](https://files.realpython.com/media/Screenshot_2024-01-12_at_9.49.31_AM.3cad959aa115.png)

Consulta de cifrado para la relación HAS

Tenga en cuenta que el resultado incluye datos para la **visita**, la relación **HAS** y el **paciente**. Esto te da más información que si solo coincides en los nodos de **visita**. Si desea ver qué médicos trataron al paciente durante la **visita**, podría agregar la siguiente relación a la consulta:

Lenguaje de consulta de cifrado

```
MATCH (p:Patient)-[h:HAS]->(v:Visit)<-[t:TREATS]-(ph:Physician)
WHERE v.id = 56
RETURN v,p,ph
```

Esta declaración `(p:Patient)-[h:HAS]->(v:Visit)<-[t:TREATS]-(ph:Physician)` le dice a Neo4j que encuentre todos los patrones en los que un `Patient` tiene una `Visit` que es tratada por un `Physician`. Si quisieras igualar todas las relaciones que entran y salen del nodo de `Visit`, podrías ejecutar esta consulta:

Lenguaje de consulta de cifrado

```
MATCH (v:Visit)-[r]-(n)
WHERE v.id = 56
RETURN r,n;
```

Tenga en cuenta ahora que la relación `[r]`, no tiene dirección con respecto a `(v:Visit)` o `(n)`. En esencia, esta declaración de coincidencia buscará todas las relaciones que entren y salgan de la `Visit` 56, junto con los nodos conectados a esas relaciones. Aquí están los resultados:

[![El cifrado coincide con todas las relaciones con un nodo](https://files.realpython.com/media/Screenshot_2024-01-12_at_10.15.30_AM.a7ebafaf83f5.png)](https://files.realpython.com/media/Screenshot_2024-01-12_at_10.15.30_AM.a7ebafaf83f5.png)

Consulta de cifrado que coincide con todas las relaciones y nodos para visitar 56

Esto te da una buena vista de todas las relaciones y nodos asociados con **Visit** 56. Piensa en lo poderosa que es esta representación. En lugar de realizar múltiples uniones SQL, como tendrías que hacer en una base de datos relacional, obtienes toda la información sobre cómo se conecta una **visita** a todo el sistema hospitalario con tres líneas cortas de Cypher.

Puedes imaginar lo poderoso que se volvería esto a medida que se agreguen más nodos y relaciones a la base de datos de gráficos. Por ejemplo, podría registrar qué enfermeras, farmacias, medicamentos o cirugías están asociadas con la **visita**. Cada relación que agregue requeriría otra unión en SQL, pero la consulta de cifrado anterior sobre **la visita** 56 se mantendría sin cambios.

Lo último que cubrirás en esta sección es cómo realizar agregaciones en Cypher. Hasta ahora, solo has consultado datos sin procesar de nodos y relaciones, pero también puedes calcular estadísticas agregadas en Cypher.

Supongamos que querías responder a la pregunta *¿Cuál es el número total de visitas y la cantidad total de facturación de las visitas cubiertas por Aetna en Texas?* Aquí está la consulta de Cypher que respondería a esta pregunta:

Lenguaje de consulta de cifrado

```
MATCH (p:Payer)<-[c:COVERED_BY]-(v:Visit)-[:AT]->(h:Hospital)
WHERE p.name = "Aetna"
AND h.state_name = "TX"
RETURN COUNT(*) as num_visits,
SUM(c.billing_amount) as total_billing_amount;
```

En esta consulta, primero coincide con todas las `Visits` que se realizan en un `Hospital` y están cubiertas por un `Payer`. A continuación, filtre a los `Payers` con una propiedad de `name` de **Aetna** y `hospitals` con un `state_name` de `TX`. Por último, `COUNT(*)` cuenta el número de patrones coincidentes, y `SUM(c.billing_amount)` le da la cantidad total de facturación. La salida se ve así:

[![Consulta agregada de cifrado](https://files.realpython.com/media/Screenshot_2024-01-12_at_10.38.24_AM.a939c19875db.png)](https://files.realpython.com/media/Screenshot_2024-01-12_at_10.38.24_AM.a939c19875db.png)

Consulta agregada de cifrado

Los resultados le dicen que hubo 198 **visitas que** coincidían con este patrón con una cantidad total de facturación de alrededor de 5.056.439 dólares.

Ahora tienes una sólida comprensión de los fundamentos de Cypher, así como de los tipos de preguntas que puedes responder. En resumen, Cypher es excelente para emparejar relaciones complicadas sin requerir una consulta detallada. Hay mucho más que puedes hacer con Neo4j y Cypher, pero el conocimiento que obtuviste en esta sección es suficiente para comenzar a construir el chatbot, y eso es lo que harás a continuación.

## Paso 4: Construir un gráfico de chatbot RAG en LangChain

Después de todo el diseño preparatorio y el trabajo de datos que has hecho hasta ahora, ¡por fin estás listo para construir tu chatbot! Es probable que te des cuenta de que, con los datos del sistema hospitalario almacenados en Neo4j y el poder de las abstracciones de LangChain, construir tu chatbot no lleva mucho trabajo. Este es un tema común en los proyectos de IA y aprendizaje automático: la mayor parte del trabajo está en el diseño, la preparación de datos y la implementación en lugar de construir la propia IA.

Antes de entrar, añade una carpeta `chatbot_api/` a tu proyecto con los siguientes archivos y carpetas:

```
./
│
├── chatbot_api/
│   │
│   ├── src/
│   │   │
│   │   ├── agents/
│   │   │   └── hospital_rag_agent.py
│   │   │
│   │   ├── chains/
│   │   │   ├── hospital_cypher_chain.py
│   │   │   └── hospital_review_chain.py
│   │   │
│   │   ├── tools/
│   │   │   └── wait_times.py
│   │
│   └── pyproject.toml
│
├── hospital_neo4j_etl/
│   │
│   ├── src/
│   │   ├── entrypoint.sh
│   │   └── hospital_bulk_csv_write.py
│   │
│   ├── Dockerfile
│   └── pyproject.toml
│
├── .env
└── docker-compose.yml
```

También querrás añadir algunas variables de entorno más a tu archivo `.env`:

```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>

NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_URI>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>

HOSPITALS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/hospitals.csv
PAYERS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/payers.csv
PHYSICIANS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/physicians.csv
PATIENTS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/patients.csv
VISITS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/visits.csv
REVIEWS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/reviews.csv

HOSPITAL_AGENT_MODEL=gpt-3.5-turbo-1106
HOSPITAL_CYPHER_MODEL=gpt-3.5-turbo-1106
HOSPITAL_QA_MODEL=gpt-3.5-turbo-0125
```

Su archivo `.env` ahora incluye variables que especifican qué LLM usará para los diferentes componentes de su chatbot. Ha especificado estos modelos como variables de entorno para que pueda cambiar fácilmente entre diferentes modelos de OpenAI sin cambiar ningún código. Sin embargo, tenga en cuenta que cada LLM podría beneficiarse de una estrategia de indicaciones única, por lo que es posible que deba modificar sus indicaciones si planea utilizar un conjunto diferente de LLM.

Ya deberías tener la carpeta `hospital_neo4j_etl/` completada, y `docker-compose.yml` y `.env` son los mismos que antes. Abre `chatbot_api/pyproject.toml` y añade las siguientes dependencias:

TOML `chatbot_api/pyproject.toml`

```
[project]
name = "chatbot_api"
version = "0.1"
dependencies = [
    "asyncio==3.4.3",
    "fastapi==0.109.0",
    "langchain==0.1.0",
    "langchain-openai==0.0.2",
    "langchainhub==0.1.14",
    "neo4j==5.14.1",
    "numpy==1.26.2",
    "openai==1.7.2",
    "opentelemetry-api==1.22.0",
    "pydantic==2.5.1",
    "uvicorn==0.25.0"
]

[project.optional-dependencies]
dev = ["black", "flake8"]
```

Ciertamente puede usar versiones más recientes de estas dependencias si están disponibles, pero tenga en cuenta cualquier característica que pueda estar obsoleta. Abra un terminal, active su entorno virtual, navegue a su carpeta `chatbot_api/` e instale dependencias desde `pyproject.toml` del proyecto:

Shell

```shell
(venv) $ python -m pip install .
```

Una vez que todo esté instalado, ¡estás listo para construir la cadena de reseñas!

### Crear una cadena vectorial Neo4j

En el paso 1, obtuve una introducción práctica a LangChain mediante la construcción de una cadena que responde a las preguntas sobre las experiencias de los pacientes utilizando sus reseñas. En esta sección, construirás una cadena similar, excepto que usarás Neo4j como índice vectorial.

[Los índices de búsqueda vectorial](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/) se publicaron como una beta pública en Neo4j 5.11. Le permiten ejecutar consultas semánticas directamente en su gráfico. Esto es muy conveniente para su chatbot porque puede almacenar las incrustaciones de revisión en el mismo lugar que los datos estructurados de su sistema hospitalario.

En LangChain, puedes usar [Neo4jVector](https://python.langchain.com/docs/integrations/vectorstores/neo4jvector) para crear incrustaciones de revisión y el retriever necesario para tu cadena. Aquí está el código para crear la cadena de reseñas:

Python `chatbot_api/src/chains/hospital_review_chain.py`

```python
import os
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

HOSPITAL_QA_MODEL = os.getenv("HOSPITAL_QA_MODEL")

neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="reviews",
    node_label="Review",
    text_node_properties=[
        "physician_name",
        "patient_name",
        "text",
        "hospital_name",
    ],
    embedding_node_property="embedding",
)

review_template = """Your job is to use patient
reviews to answer questions about their experience at a hospital. Use
the following context to answer questions. Be as detailed as possible, but
don't make up any information that's not from the context. If you don't know
an answer, say you don't know.
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=review_template)
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [review_system_prompt, review_human_prompt]

review_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

reviews_vector_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=HOSPITAL_QA_MODEL, temperature=0),
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(k=12),
)
reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt
```

En las líneas 1 a 11, importa las dependencias necesarias para construir su cadena de revisión con Neo4j. En la línea 13, carga el nombre del modelo de chat que utilizará para la cadena de revisión y lo almacena en `HOSPITAL_QA_MODEL`. Las líneas 15 a 29 crean el índice vectorial en Neo4j. Aquí hay un desglose de cada parámetro:

- `embedding`: El modelo utilizado para crear las incrustaciones: está utilizando  `OpenAIEmeddings()`en este ejemplo.
- `url`, `username`, y `password`: Tus credenciales de instancia de Neo4j.
- `index_name`: El nombre dado a su índice vectorial.
- `node_label`: El nodo para el que crear incrustaciones.
- `text_node_properties`: Las propiedades del nodo que se incluirán en la incrustación.
- `embedding_node_property`: El nombre de la propiedad del nodo de incrustación.

Una vez que se ejecute `Neo4jVector.from_existing_graph()`, verá que cada nodo de **revisión** en Neo4j tiene una propiedad **de incrustación** que es una representación vectorial de las propiedades **physician_name**, **patient_name**, **text** y **hospital_name**. Esto le permite responder a preguntas como *¿Qué hospitales han tenido críticas positivas?* También permite que el LLM le diga qué paciente y médico escribieron reseñas que coinciden con su pregunta.

Las líneas 31 a 50 crean la plantilla de aviso para su cadena de revisión de la misma manera que lo hizo en el [paso 1.](https://realpython.com/build-llm-rag-chatbot-with-langchain/#prompt-templates)

Lastly, lines 52 to 57 create your reviews vector chain using a Neo4j vector index retriever that returns 12 reviews embeddings from a similarity search. By setting `chain_type` to `"stuff"` in `.from_chain_type()`, you’re telling the chain to pass all 12 reviews to the prompt. You can explore other chain types in [LangChain’s documentation on chains](https://python.langchain.com/docs/modules/chains).

Por último, las líneas 52 a 57 crean su cadena vectorial de reseñas utilizando un recuperador de índice vectorial Neo4j que devuelve 12 incrustaciones de reseñas de una búsqueda de similitud. Al establecer `chain_type` en `"suff"` en `.from_chain_type()`, le estás diciendo a la cadena que pase las 12 revisiones al mensaje. Puedes explorar otros tipos de cadenas en  [LangChain’s documentation on chains](https://python.langchain.com/docs/modules/chains).

Estás listo para probar tu nueva cadena de reseñas. Vaya al directorio raíz de su proyecto, inicie un intérprete de Python y ejecute los siguientes comandos:

```python
>>> import dotenv
>>> dotenv.load_dotenv()
True

>>> from chatbot_api.src.chains.hospital_review_chain import (
...     reviews_vector_chain
... )

>>> query = """What have patients said about hospital efficiency?
...         Mention details from specific reviews."""

>>> response = reviews_vector_chain.invoke(query)

>>> response.get("result")
"Patients have mentioned different aspects of hospital efficiency in their
reviews. In Kevin Cox's review of Wallace-Hamilton hospital, he mentioned
that the hospital staff was efficient. However, he also mentioned a lack of
personalized attention and communication, which left him feeling neglected.
This suggests that while the hospital may have been efficient in terms of
completing tasks and providing services, they may have lacked in terms of
individualized care and communication with patients.
On the other hand, Beverly Johnson's review of Brown Inc. hospital mentioned
that the hospital had a modern feel and the staff was attentive. However,
she also mentioned that the bureaucratic procedures for check-in and
discharge were cumbersome. This suggests that while the hospital may have
been efficient in terms of its facilities and staff attentiveness, the
administrative processes may have been inefficient and caused inconvenience
for patients. It is important to note that the specific reviews do not
provide a comprehensive picture of hospital efficiency, as they focus on
specific aspects of the hospital experience."
```

En este bloque, importa `dotenv` y carga variables de entorno desde `.env`. A continuación, importa `reviews_vector_chain` de `hospital_review_chain` y la invoca con una pregunta sobre la eficiencia del hospital. La respuesta de su cadena puede no ser idéntica a esta, pero el LLM debería devolver un buen resumen detallado, como usted le ha dicho.

En este ejemplo, observe cómo se mencionan los nombres específicos de los pacientes y los hospitales en la respuesta. Esto sucede porque ha insertado los nombres de los hospitales y los pacientes junto con el texto de la revisión, por lo que el LLM puede usar esta información para responder preguntas.

**Note**: Before moving on, you should play around with `reviews_vector_chain` to see how it responds to different queries. Do the responses seem correct? How might you evaluate the quality of `reviews_vector_chain`? You won’t learn how to evaluate RAG systems in this tutorial, but you can look at this [comprehensive Python example with MLFlow](https://mlflow.org/docs/latest/llms/rag/notebooks/mlflow-e2e-evaluation.html) to get a feel for how it’s done.

> **Nota**: Antes de seguir adelante, deberías jugar con `reviews_vector_chain` para ver cómo responde a las diferentes consultas. ¿Las respuestas parecen correctas? ¿Cómo podrías evaluar la calidad de `reviews_vector_chain`? No aprenderás a evaluar los sistemas RAG en este tutorial, pero puedes ver [este ejemplo completo de Python con MLFlow](https://mlflow.org/docs/latest/llms/rag/notebooks/mlflow-e2e-evaluation.html) para tener una idea de cómo se hace.

A continuación, creará la cadena de generación de Cypher que utilizará para responder a las consultas sobre los datos estructurados del sistema hospitalario.

### Crea una cadena de cifrado Neo4j

Como vio en el paso 2, su cadena de cifrado Neo4j aceptará la consulta de lenguaje natural de un usuario, convertirá la consulta de lenguaje natural en una consulta de cifrado, ejecutará la consulta de cifrado en Neo4j y utilizará los resultados de la consulta de cifrado para responder a la consulta del usuario. Aprovecharás `GraphCypherQAChain` de LangChain para esto.

> **Nota**: Cada vez que permita a los usuarios consultar una base de datos, como lo hará con su cadena de cifrado, debe asegurarse de que solo tengan los permisos necesarios. Las credenciales de Neo4j que está utilizando en este proyecto permiten a los usuarios leer, escribir, actualizar y eliminar datos de su base de datos.

Si estuvieras construyendo esta aplicación para un proyecto del mundo real, querrías crear credenciales que restrinjan los permisos de tu usuario a solo lectura, impidiendo que escriban o eliminen datos valiosos.

Usar LLM para generar consultas de cifrado precisas puede ser un desafío, especialmente si tienes un gráfico complicado. Debido a esto, se requiere mucha ingeniería rápida para mostrar la estructura de su gráfico y los casos de uso de la consulta al LLM. [Ajustar](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) un LLM para generar consultas también es una opción, pero esto requiere datos seleccionados y etiquetados manualmente.

Para comenzar a crear su cadena de generación de Cypher, importe dependencias e instacie un AnNeo4jGraph:

Python `chatbot_api/src/chains/hospital_cypher_chain.py`

```python
import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

HOSPITAL_QA_MODEL = os.getenv("HOSPITAL_QA_MODEL")
HOSPITAL_CYPHER_MODEL = os.getenv("HOSPITAL_CYPHER_MODEL")

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

graph.refresh_schema()
```

El objeto `Neo4jGraph` es una envoltura de LangChain que permite a los LLM ejecutar consultas en su instancia Neo4j. Usted crea una instancia del `graph` utilizando sus credenciales de Neo4j y llama a `graph.refresh_schema()` para sincronizar cualquier cambio reciente en su instancia.

El siguiente y más importante componente de su cadena de generación de Cypher es la plantilla rápida. Así es como se ve:

Python `chatbot_api/src/chains/hospital_cypher_chain.py`

```python
# ...

cypher_generation_template = """
Task:
Generate Cypher query for a Neo4j graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything other than
for you to construct a Cypher statement. Do not include any text except
the generated Cypher statement. Make sure the direction of the relationship is
correct in your queries. Make sure you alias both entities and relationships
properly. Do not run any queries that would add to or delete from
the database. Make sure to alias all statements that follow as with
statement (e.g. WITH v as visit, c.billing_amount as billing_amount)
If you need to divide numbers, make sure to
filter the denominator to be non zero.

Examples:
# Who is the oldest patient and how old are they?
MATCH (p:Patient)
RETURN p.name AS oldest_patient,
       duration.between(date(p.dob), date()).years AS age
ORDER BY age DESC
LIMIT 1

# Which physician has billed the least to Cigna
MATCH (p:Payer)<-[c:COVERED_BY]-(v:Visit)-[t:TREATS]-(phy:Physician)
WHERE p.name = 'Cigna'
RETURN phy.name AS physician_name, SUM(c.billing_amount) AS total_billed
ORDER BY total_billed
LIMIT 1

# Which state had the largest percent increase in Cigna visits
# from 2022 to 2023?
MATCH (h:Hospital)<-[:AT]-(v:Visit)-[:COVERED_BY]->(p:Payer)
WHERE p.name = 'Cigna' AND v.admission_date >= '2022-01-01' AND
v.admission_date < '2024-01-01'
WITH h.state_name AS state, COUNT(v) AS visit_count,
     SUM(CASE WHEN v.admission_date >= '2022-01-01' AND
     v.admission_date < '2023-01-01' THEN 1 ELSE 0 END) AS count_2022,
     SUM(CASE WHEN v.admission_date >= '2023-01-01' AND
     v.admission_date < '2024-01-01' THEN 1 ELSE 0 END) AS count_2023
WITH state, visit_count, count_2022, count_2023,
     (toFloat(count_2023) - toFloat(count_2022)) / toFloat(count_2022) * 100
     AS percent_increase
RETURN state, percent_increase
ORDER BY percent_increase DESC
LIMIT 1

# How many non-emergency patients in North Carolina have written reviews?
MATCH (r:Review)<-[:WRITES]-(v:Visit)-[:AT]->(h:Hospital)
WHERE h.state_name = 'NC' and v.admission_type <> 'Emergency'
RETURN count(*)

String category values:
Test results are one of: 'Inconclusive', 'Normal', 'Abnormal'
Visit statuses are one of: 'OPEN', 'DISCHARGED'
Admission Types are one of: 'Elective', 'Emergency', 'Urgent'
Payer names are one of: 'Cigna', 'Blue Cross', 'UnitedHealthcare', 'Medicare',
'Aetna'

A visit is considered open if its status is 'OPEN' and the discharge date is
missing.
Use abbreviations when
filtering on hospital states (e.g. "Texas" is "TX",
"Colorado" is "CO", "North Carolina" is "NC",
"Florida" is "FL", "Georgia" is "GA", etc.)

Make sure to use IS NULL or IS NOT NULL when analyzing missing properties.
Never return embedding properties in your queries. You must never include the
statement "GROUP BY" in your query. Make sure to alias all statements that
follow as with statement (e.g. WITH v as visit, c.billing_amount as
billing_amount)
If you need to divide numbers, make sure to filter the denominator to be non
zero.

The question is:
{question}
"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)
```

Lea atentamente el contenido de `cypher_generation_template`. Observe cómo está proporcionando al LLM instrucciones muy específicas sobre lo que debe y no debe hacer al generar consultas de cifrado. Lo más importante es que está mostrando el LLM la estructura de su gráfico con el parámetro del `schema`, algunas consultas de ejemplo y los valores categóricos de algunas propiedades de nodo.

Todos los detalles que proporciona en su plantilla de solicitud mejoran la posibilidad de LLM de generar una consulta de cifrado correcta para una pregunta determinada. Si tienes curiosidad por saber lo necesario que es todo este detalle, intenta crear tu propia plantilla de aviso con el menor número de detalles posible. Luego ejecute las preguntas a través de su cadena Cypher y vea si genera correctamente las consultas Cypher.

A partir de ahí, puede actualizar de forma iterativa su plantilla de solicitud para corregir las consultas que el LLM tiene dificultades para generar, pero asegúrese de que también está al tanto de la cantidad de tokens de entrada que está utilizando. Al igual que con su cadena de revisión, querrá un sistema sólido para evaluar las plantillas de solicitud y la corrección de las consultas de cifrado generadas por su cadena. Sin embargo, como verás, la plantilla que tienes arriba es un gran punto de partida.

> **Nota**: La plantilla de aviso anterior proporciona al LLM cuatro ejemplos de consultas de cifrado válidas para su gráfico. Darle al LLM algunos ejemplos y luego pedirle que realice una tarea se conoce como [prompting de pocos disparos](https://realpython.com/practical-prompt-engineering/#start-engineering-your-prompts), y es una técnica simple pero poderosa para mejorar la precisión de la generación.

Sin embargo, el aviso de pocos disparos podría no ser suficiente para la generación de consultas de cifrado, especialmente si tiene un gráfico complicado. Una forma de mejorar esto es crear una base de datos vectorial que incruste preguntas/consultas de usuarios de ejemplo y almacene sus consultas de cifrado correspondientes como metadatos.

Cuando un usuario hace una pregunta, se inyectan consultas de cifrado de preguntas semánticamente similares en el mensaje, proporcionando al LLM los ejemplos más relevantes necesarios para responder a la pregunta actual.

A continuación, define la plantilla de aviso para el componente de pregunta y respuesta de su cadena. Esta plantilla le dice al LLM que utilice los resultados de la consulta de cifrado para generar una respuesta con un buen formato a la consulta del usuario:

Python `chatbot_api/src/chains/hospital_cypher_chain.py`

```python
# ...

qa_generation_template = """You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response. The
query results section contains the results of a Cypher query that was
generated based on a user's natural language question. The provided
information is authoritative, you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question.

Query Results:
{context}

Question:
{question}

If the provided information is empty, say you don't know the answer.
Empty information looks like this: []

If the information is not empty, you must provide an answer using the
results. If the question involves a time duration, assume the query
results are in units of days unless otherwise specified.

When names are provided in the query results, such as hospital names,
beware  of any names that have commas or other punctuation in them.
For instance, 'Jones, Brown and Murray' is a single hospital name,
not multiple hospitals. Make sure you return any list of names in
a way that isn't ambiguous and allows someone to tell what the full
names are.

Never say you don't have the right information if there is data in
the query results. Always use the data in the query results.

Helpful Answer:
"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)
```

Esta plantilla requiere mucho menos detalle que su plantilla de generación de Cypher, y solo debería tener que modificarla si desea que el LLM responda de manera diferente, o si se da cuenta de que no está utilizando los resultados de la consulta como desea. El último paso para crear su cadena Cypher es crear una instancia de un objeto `GraphCypherQAChain`:

Python `chatbot_api/src/chains/hospital_cypher_chain.py`

```python
# ...

hospital_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(model=HOSPITAL_CYPHER_MODEL, temperature=0),
    qa_llm=ChatOpenAI(model=HOSPITAL_QA_MODEL, temperature=0),
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
)
```

Aquí hay un desglose de los parámetros utilizados en `GraphCypherQAChain.from_llm()`:

- `cypher_llm`: El LLM utilizado para generar consultas de cifrado.
- `qa_llm`: El LLM utilizado para generar una respuesta dados los resultados de la consulta de Cypher.
- `graph`: El objeto `Neo4jGraph` que se conecta a su instancia Neo4j.
- `verbose`: Se debe imprimir si los pasos intermedios que realiza su cadena.
- `qa_prompt`: La plantilla de aviso para responder a preguntas/preguntas.
- `cypher_prompt`: La plantilla de aviso para generar consultas de cifrado.
- `validate_cypher`: Si es cierto, la consulta de cifrado se inspeccionará en busca de errores y se corregirá antes de ejecutarla. Tenga en cuenta que esto no garantiza que la consulta de Cypher sea válida. En su lugar, corrige errores de sintaxis simples que son fácilmente detectables usando [expresiones regulares](https://realpython.com/regex-python/).
- `top_k`: El número de resultados de la consulta que se incluirán en `qa_prompt`.

¡La cadena de generación de Cypher de su sistema hospitalario está lista para usar! Funciona de la misma manera que tu cadena de reseñas. Navegue hasta el directorio de su proyecto e inicie una nueva sesión de intérprete de Python, luego pruébelo:

```python
>>> import dotenv
>>> dotenv.load_dotenv()
True

>>> from chatbot_api.src.chains.hospital_cypher_chain import (
... hospital_cypher_chain
... )

>>> question = """What is the average visit duration for
... emergency visits in North Carolina?"""
>>> response = hospital_cypher_chain.invoke(question)


> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (v:Visit)-[:AT]->(h:Hospital)
WHERE h.state_name = 'NC' AND v.admission_type = 'Emergency'
AND v.status = 'DISCHARGED'
WITH v, duration.between(date(v.admission_date),
date(v.discharge_date)).days AS visit_duration
RETURN AVG(visit_duration) AS average_visit_duration
Full Context:
[{'average_visit_duration': 15.072972972972991}]

> Finished chain.

>>> response.get("result")
'The average visit duration for emergency visits in North
Carolina is 15.07 days.'
```

Después de cargar las variables de entorno, importar `hospital_cypher_chain` e invocarla con una pregunta, puede ver los pasos que toma su cadena para responder a la pregunta. Tómese un segundo para apreciar algunos logros que su cadena hizo al generar la consulta de Cypher:

- La generación Cypher LLM entendió la relación entre las visitas y los hospitales a partir del esquema gráfico proporcionado.
- A pesar de que le preguntaste sobre **Carolina del Norte**, el LLM sabía por el aviso de usar la abreviatura estatal **NC**.
- El LLM sabía que las propiedades de **tipo de admisión** solo tienen la primera letra en mayúscula, mientras que las propiedades **de estado** son todas mayúsculas.
- La generación de control de calidad LLM sabía por su solicitud que los resultados de la consulta estaban en unidades de días.

Puedes experimentar con todo tipo de preguntas sobre el sistema hospitalario. Por ejemplo, aquí hay una pregunta relativamente difícil de convertir a Cypher:

```python
>>> question = """Which state had the largest percent increase
...            in Medicaid visits from 2022 to 2023?"""
>>> response = hospital_cypher_chain.invoke(question)


> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (h:Hospital)<-[:AT]-(v:Visit)-[:COVERED_BY]->(p:Payer)
WHERE p.name = 'Medicaid' AND v.admission_date >= '2022-01-01'
AND v.admission_date < '2024-01-01'
WITH h.state_name AS state, COUNT(v) AS visit_count,
     SUM(CASE WHEN v.admission_date >= '2022-01-01'
     AND v.admission_date < '2023-01-01' THEN 1 ELSE 0 END) AS count_2022,
     SUM(CASE WHEN v.admission_date >= '2023-01-01'
     AND v.admission_date < '2024-01-01' THEN 1 ELSE 0 END) AS count_2023
WITH state, visit_count, count_2022, count_2023,
     (toFloat(count_2023) - toFloat(count_2022)) / toFloat(count_2022) * 100
     AS percent_increase
RETURN state, percent_increase
ORDER BY percent_increase DESC
LIMIT 1
Full Context:
[{'state': 'TX', 'percent_increase': 8.823529411764707}]

> Finished chain.

>>> response.get("result")
'The state with the largest percent increase in Medicaid visits
from 2022 to 2023 is Texas (TX), with a percent increase of 8.82%.'
```

Para responder a la pregunta *¿Qué estado tuvo el mayor aumento porcentual en las visitas de Medicaid de 2022 a 2023?*, el LLM tuvo que generar una consulta de Cipher bastante detallada que involucraba múltiples nodos, relaciones y filtros. Sin embargo, fue capaz de llegar a la respuesta correcta.

La última capacidad que tu chatbot necesita es responder preguntas sobre los tiempos de espera, y eso es lo que cubrirás a continuación.

### Crear funciones de tiempo de espera

Esta última capacidad que su chatbot necesita es responder preguntas sobre los tiempos de espera del hospital. Como se discutió anteriormente, su organización no almacena datos de tiempo de espera en ninguna parte, por lo que su chatbot tendrá que obtenerlos de una fuente externa. Escribirás dos funciones para esto: una que simula encontrar el tiempo de espera actual en un hospital, y otra que encuentra el hospital con el tiempo de espera más corto.

> **Nota**: El propósito de crear funciones de tiempo de espera es mostrarle que los agentes de LangChain pueden ejecutar código Python arbitrario, no solo cadenas u otros métodos de LangChain. Esta capacidad es extremadamente valiosa porque significa que, en teoría, podrías crear un agente para hacer casi cualquier cosa que se pueda expresar en código.

Comience por definir las funciones para obtener los tiempos de espera actuales en un hospital:

Python `chatbot_api/src/tools/wait_times.py`

```python
import os
from typing import Any
import numpy as np
from langchain_community.graphs import Neo4jGraph

def _get_current_hospitals() -> list[str]:
    """Fetch a list of current hospital names from a Neo4j database."""
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )

    current_hospitals = graph.query(
        """
        MATCH (h:Hospital)
        RETURN h.name AS hospital_name
        """
    )

    return [d["hospital_name"].lower() for d in current_hospitals]

def _get_current_wait_time_minutes(hospital: str) -> int:
    """Get the current wait time at a hospital in minutes."""
    current_hospitals = _get_current_hospitals()

    if hospital.lower() not in current_hospitals:
        return -1

    return np.random.randint(low=0, high=600)


def get_current_wait_times(hospital: str) -> str:
    """Get the current wait time at a hospital formatted as a string."""
    wait_time_in_minutes = _get_current_wait_time_minutes(hospital)

    if wait_time_in_minutes == -1:
        return f"Hospital '{hospital}' does not exist."

    hours, minutes = divmod(wait_time_in_minutes, 60)

    if hours > 0:
        return f"{hours} hours {minutes} minutes"
    else:
        return f"{minutes} minutes"
```

La primera función que define es `_get_current_hospitals()`, que devuelve una lista de nombres de hospitales de su base de datos Neo4j. Luego, `_get_current_wait_time_minutes()` toma un nombre de hospital como entrada. Si el nombre del hospital no es válido, `_get_current_wait_time_minutes()` devuelve -1. Si el nombre del hospital es válido, `_get_current_wait_time_minutes()` devuelve un entero aleatorio entre 0 y 600 simulando un tiempo de espera en minutos.

A continuación, define `get_current_wait_times()`, que es una envoltura alrededor de `_get_current_wait_time_minutes()` que devuelve el tiempo de espera formateado como una cadena.

Puedes usar `_get_current_wait_time_minutes()` para definir una segunda función que encuentre el hospital con el tiempo de espera más corto:

Python `chatbot_api/src/tools/wait_times.py`

```python
# ...

def get_most_available_hospital(_: Any) -> dict[str, float]:
    """Find the hospital with the shortest wait time."""
    current_hospitals = _get_current_hospitals()

    current_wait_times = [
        _get_current_wait_time_minutes(h) for h in current_hospitals
    ]

    best_time_idx = np.argmin(current_wait_times)
    best_hospital = current_hospitals[best_time_idx]
    best_wait_time = current_wait_times[best_time_idx]

    return {best_hospital: best_wait_time}
```

Aquí, define `get_most_available_hospital()` que llama a `_get_current_wait_time_minutes()` en cada hospital y devuelve el hospital con el tiempo de espera más corto. Observe cómo `get_most_available_hospital()` tiene una [throwaway input](https://realpython.com/python-double-underscore/#public-interfaces-and-naming-conventions-in-python)_. Esto será requerido más tarde por su agente porque está diseñado para pasar entradas a las funciones.

Así es como usas `get_current_wait_times()` y `get_most_available_hospital()`:

Python

```python
>>> import dotenv
>>> dotenv.load_dotenv()
True

>>> from chatbot_api.src.tools.wait_times import (
...     get_current_wait_times,
...     get_most_available_hospital,
... )

>>> get_current_wait_times("Wallace-Hamilton")
'1 hours 35 minutes'

>>> get_current_wait_times("fake hospital")
"Hospital 'fake hospital' does not exist."

>>> get_most_available_hospital(None)
{'cunningham and sons': 24}
```

Después de cargar las variables de entorno, llama a `get_current_wait_times("Wallace-Hamilton")` que devuelve el tiempo de espera actual en minutos en el hospital **Wallace-Hamilton**. Cuando intentas `get_current_wait_times("fake hospital")`, obtienes una cadena que dice que el hospital falso no existe en la base de datos.

Por último, `get_most_available_hospital()` devuelve un diccionario que almacena el tiempo de espera para el hospital con el tiempo de espera más corto en minutos. A continuación, crearás un agente que utiliza estas funciones, junto con la cadena Cypher y de revisión, para responder a preguntas arbitrarias sobre el sistema hospitalario.

### Crear el agente de chatbot

Date una palmadita en la espalda si has llegado hasta aquí. Has cubierto mucha información, y finalmente estás listo para armarlo todo y reunir al agente que servirá como tu chatbot. Dependiendo de la consulta que le dé, su agente debe decidir entre su cadena de cifrado, la cadena de revisiones y las funciones de tiempos de espera.

Comience cargando las dependencias de su agente, leyendo el nombre del modelo del agente desde una variable de entorno y cargando una plantilla de solicitud desde [LangChain Hub](https://smith.langchain.com/hub):

Python `chatbot_api/src/agents/hospital_rag_agent.py`

```
import os
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub
from chains.hospital_review_chain import reviews_vector_chain
from chains.hospital_cypher_chain import hospital_cypher_chain
from tools.wait_times import (
    get_current_wait_times,
    get_most_available_hospital,
)

HOSPITAL_AGENT_MODEL = os.getenv("HOSPITAL_AGENT_MODEL")

hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")
```

Observe cómo está importando `reviews_vector_chain`, `hospital_cypher_chain`, `get_current_wait_times()` y `get_most_available_hospital()`. Su agente los utilizará directamente como herramientas. `HOSPITAL_AGENT_MODEL` es el LLM que actuará como el cerebro de su agente, decidiendo a qué herramientas llamar y qué entradas pasarlas.

En lugar de definir tu propio aviso para el agente, lo que sin duda puedes hacer, cargas un mensaje predefinido de LangChain Hub. El centro de LangChain le permite cargar, navegar, extraer, probar y administrar solicitudes. En este caso, el mensaje predeterminado para los agentes de funciones OpenAI funciona muy bien.

A continuación, define una lista de herramientas que su agente puede usar:

Python `chatbot_api/src/agents/hospital_rag_agent.py`

```python
# ...

tools = [
    Tool(
        name="Experiences",
        func=reviews_vector_chain.invoke,
        description="""Useful when you need to answer questions
        about patient experiences, feelings, or any other qualitative
        question that could be answered about a patient using semantic
        search. Not useful for answering objective questions that involve
        counting, percentages, aggregations, or listing facts. Use the
        entire prompt as input to the tool. For instance, if the prompt is
        "Are patients satisfied with their care?", the input should be
        "Are patients satisfied with their care?".
        """,
    ),
    Tool(
        name="Graph",
        func=hospital_cypher_chain.invoke,
        description="""Useful for answering questions about patients,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?".
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_times,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. Do not pass the word "hospital"
        as input, only the hospital name itself. For example, if the prompt
        is "What is the current wait time at Jordan Inc Hospital?", the
        input should be "Jordan Inc".
        """,
    ),
    Tool(
        name="Availability",
        func=get_most_available_hospital,
        description="""
        Use when you need to find out which hospital has the shortest
        wait time. This tool does not have any information about aggregate
        or historical wait times. This tool returns a dictionary with the
        hospital name as the key and the wait time in minutes as the value.
        """,
    ),
]
```

Su agente tiene cuatro herramientas disponibles: **Experiencias***, **Gráfico*, **Esperas** y **Disponibilidad**. Las herramientas **Experiencias** y **Gráficos** llaman a `.invoke()` desde sus respectivas cadenas, mientras que las `Waits` y la `Availability` llaman a las funciones de tiempo de espera que definiste. Tenga en cuenta que muchas de las descripciones de las herramientas tienen pocas indicaciones de disparo, diciéndole al agente cuándo debe usar la herramienta y proporcionándole un ejemplo de qué entradas debe pasar.

Al igual que con las cadenas, una buena ingeniería rápida es crucial para el éxito de su agente. Tienes que describir claramente cada herramienta y cómo usarla para que tu agente no se confunda con una consulta.

El último paso es crear una instancia de su agente:

Python `chatbot_api/src/agents/hospital_rag_agent.py`

```python
# ...

chat_model = ChatOpenAI(
    model=HOSPITAL_AGENT_MODEL,
    temperature=0,
)

hospital_rag_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=hospital_agent_prompt,
    tools=tools,
)

hospital_rag_agent_executor = AgentExecutor(
    agent=hospital_rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
```

Primero inicializas un objeto `ChatOpenAI` usando `HOSPITAL_AGENT_MODEL` como LLM. A continuación, crea un agente de funciones OpenAI con `create_openai_functions_agent()`. Esto crea un agente que ha sido diseñado por OpenAI para pasar entradas a las funciones. Lo hace devolviendo objetos JSON que almacenan las entradas de la función y su valor correspondiente.

Para crear el tiempo de ejecución del agente, pasas tu agente y tus herramientas a `AgentExecutor`. Establecer `return_intermediate_steps` y `verbose` en true le permite ver el proceso de pensamiento del agente y las herramientas que llama.

Con eso, has completado la construcción del agente del sistema hospitalario. Para probarlo, tendrás que navegar a la carpeta `chatbot_api/src/` e iniciar una nueva sesión de REPL desde allí.

> **Nota**: Esto es necesario porque configuras importaciones relativas en **hospital_rag_agent.py** que luego se ejecutarán dentro de un contenedor Docker. Por ahora, significa que tendrás que iniciar tu intérprete de Python solo después de navegar a `chatbot_api/src/` para que las importaciones funcionen.

Ahora puedes probar tu agente del sistema hospitalario en tu línea de comandos:

Python

```python
>>> import dotenv
>>> dotenv.load_dotenv()
True

>>> from agents.hospital_rag_agent import hospital_rag_agent_executor

>>> response = hospital_rag_agent_executor.invoke(
...     {"input": "What is the wait time at Wallace-Hamilton?"}
... )

> Entering new AgentExecutor chain...

Invoking: `Waits` with `Wallace-Hamilton`

54The current wait time at Wallace-Hamilton is 54 minutes.

> Finished chain.

>>> response.get("output")
'The current wait time at Wallace-Hamilton is 54 minutes.'

>>> response = hospital_rag_agent_executor.invoke(
...     {"input": "Which hospital has the shortest wait time?"}
... )

> Entering new AgentExecutor chain...

Invoking: `Availability` with `shortest wait time`

{'smith, edwards and obrien': 2}The hospital with the shortest
wait time is Smith, Edwards and O'Brien, with a wait time of 2 minutes.

> Finished chain.

>>> response.get("output")
"The hospital with the shortest wait time is Smith, Edwards
and O'Brien, with a wait time of 2 minutes."
```

Después de cargar las variables de entorno, le preguntas al agente sobre los tiempos de espera. Puedes ver exactamente lo que está haciendo en respuesta a cada una de tus preguntas. Por ejemplo, cuando preguntas *"¿Cuál es el tiempo de espera en Wallace-Hamilton?"*, invoca la herramienta de **Wait** y pasa a **Wallace-Hamilton** como entrada. Esto significa que el agente está llamando a `get_current_wait_times("Wallace-Hamilton")`, observando el valor de retorno y utilizando el valor de retorno para responder a su pregunta.

Para ver todas las capacidades de los agentes, puede hacerle preguntas sobre las experiencias de los pacientes que requieren revisiones de los pacientes para responder:

Python

```python
>>> response = hospital_rag_agent_executor.invoke(
...     {
...         "input": (
...             "What have patients said about their "
...             "quality of rest during their stay?"
...         )
...     }
... )

> Entering new AgentExecutor chain...

Invoking: `Experiences` with `What have patients said about their quality of
rest during their stay?`

{'query': 'What have patients said about their quality of rest during their
stay?','result': "Patients have mentioned that the constant interruptions
for routine checks and the noise level at night were disruptive and made
it difficult for them to get a good night's sleep during their stay.
Additionally, some patients have complained about uncomfortable beds
affecting their quality of rest."}Patients have mentioned that the
constant interruptions for routine checks and the noise level at night
were disruptive and made it difficult for them to get a good night's sleep
during their stay. Additionally, some patients have complained about
uncomfortable beds affecting their quality of rest.

> Finished chain.

>>> response.get("output")
"Patients have mentioned that the constant interruptions for routine checks
and the noise level at night were disruptive and made it difficult for them
to get a good night's sleep during their stay. Additionally, some patients
have complained about uncomfortable beds affecting their quality of rest."
```

Observe aquí cómo nunca menciona explícitamente las reseñas o experiencias en su pregunta. El agente sabe, según la descripción de la herramienta, que necesita invocar **Experiencias**. Por último, puede hacerle al agente una pregunta que requiera una consulta de Cypher para responder:

```python
>>> response = hospital_rag_agent_executor.invoke(
...     {
...         "input": (
...             "Which physician has treated the "
...             "most patients covered by Cigna?"
...         )
...     }
... )

> Entering new AgentExecutor chain...

Invoking: `Graph` with `Which physician has treated the most patients
covered by Cigna?`

> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (phy:Physician)-[:TREATS]->(v:Visit)-[:COVERED_BY]->(p:Payer)
WHERE p.name = 'Cigna'
WITH phy, COUNT(DISTINCT v) AS patient_count
RETURN phy.name AS physician_name, patient_count
ORDER BY patient_count DESC
LIMIT 1
Full Context:
[{'physician_name': 'Renee Brown', 'patient_count': 10}]

> Finished chain.
{'query': 'Which physician has treated the most patients covered by Cigna?',
'result': 'The physician who has treated the most patients covered by Cigna
is Dr. Renee Brown. She has treated a total of 10 patients.'}The
physician who has treated the most patients covered by Cigna is Dr. Renee
Brown. She has treated a total of 10 patients.

> Finished chain.

>>> response.get("output")
'The physician who has treated the most patients covered by
Cigna is Dr. Renee Brown.
She has treated a total of 10 patients.'
```

Su agente tiene una notable capacidad para saber qué herramientas usar y qué entradas pasar en función de su consulta. Este es tu chatbot en pleno funcionamiento. Tiene el potencial de responder a todas las preguntas que sus partes interesadas puedan hacer en función de los requisitos dados, y parece estar haciendo un gran trabajo hasta ahora.

A medida que le hagas más preguntas a tu chatbot, es casi seguro que te encontrarás con situaciones en las que llama a la herramienta equivocada o genera una respuesta incorrecta. Si bien la modificación de sus indicaciones puede ayudar a abordar las respuestas incorrectas, a veces puede modificar su consulta de entrada para ayudar a su chatbot. Echa un vistazo a este ejemplo:

```python
>>> response = hospital_rag_agent_executor.invoke(
...     {"input": "Show me reviews written by patient 7674."}
... )

> Entering new AgentExecutor chain...

Invoking: `Experiences` with `Show me reviews written by patient 7674.`

{'query': 'Show me reviews written by patient 7674.', 'result': 'I\'m sorry,
but there are no reviews provided by a patient with the identifier "7674" in
the context given. If you have any other questions or need information about
the reviews provided, feel free to ask.'}I'm sorry, but there are no reviews
provided by a patient with the identifier "7674" in the context given. If
you have any other questions or need information about the reviews provided,
feel free to ask.

> Finished chain.

>>> response.get("output")
'I\'m sorry, but there are no reviews provided by a patient with the identifier
"7674" in the context given. If you have any other questions or need information
about the reviews provided, feel free to ask.'
```

En este ejemplo, le pides al agente que te muestre las reseñas escritas por el paciente 7674. Tu agente invoca `Experiences` y no encuentra la respuesta que estás buscando. Si bien puede ser posible encontrar la respuesta utilizando la búsqueda de vectores semánticos, puede obtener una respuesta exacta generando una consulta de Cypher para buscar reseñas correspondientes al ID del paciente 7674. Para ayudar a su agente a entender esto, puede agregar detalles adicionales a su consulta:

```python
>>> response = hospital_rag_agent_executor.invoke(
...     {
...         "input": (
...             "Query the graph database to show me "
...             "the reviews written by patient 7674"
...         )
...     }
... )

> Entering new AgentExecutor chain...

Invoking: `Graph` with `Show me reviews written by patient 7674`

> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (p:Patient {id: 7674})-[:HAS]->(v:Visit)-[:WRITES]->(r:Review)
RETURN r.text AS review_written

Full Context:
[{'review_written': 'The hospital provided exceptional care,
but the billing process was confusing and frustrating. Clearer
communication about costs would have been appreciated.'}]

> Finished chain.
{'query': 'Show me reviews written by patient 7674', 'result': 'Here
is a review written by patient 7674: "The hospital provided exceptional
care, but the billing process was confusing and frustrating. Clearer
communication about costs would have been appreciated."'}Patient 7674
wrote the following review: "The hospital provided exceptional
care, but the billing process was confusing and frustrating.
Clearer communication about costs would have been appreciated."

> Finished chain.

>>> response.get("output")
'Patient 7674 wrote the following review: "The hospital provided exceptional
care, but the billing process was confusing and frustrating. Clearer
communication about costs would have been appreciated."'
```

Aquí, le dice explícitamente a su agente que desea consultar la base de datos de gráficos, que invoca correctamente a `Graph` para encontrar la revisión que coincide con el ID de paciente 7674. Proporcionar más detalles en sus consultas como esta es una forma simple pero efectiva de guiar a su agente cuando está invocando claramente las herramientas equivocadas.

Al igual que con sus revisiones y la cadena de Cypher, antes de colocar esto frente a las partes interesadas, le gustaría llegar a un marco para evaluar a su agente. La funcionalidad principal que le gustaría evaluar es la capacidad del agente para llamar a las herramientas correctas con las entradas correctas, y su capacidad para comprender e interpretar los resultados de las herramientas a las que llama.

En el paso final, aprenderá a implementar su agente del sistema hospitalario con FastAPI y Streamlit. Esto hará que su agente sea accesible para cualquier persona que llame al punto final de la API o interactúe con la interfaz de usuario de Streamlit.

## Paso 5: Implementar el agente LangChain[](https://realpython.com/build-llm-rag-chatbot-with-langchain/#step-5-deploy-the-langchain-agent "Enlace permanente")

Por fin, tienes un agente de LangChain en funcionamiento que sirve como chatbot del sistema hospitalario. Lo último que tienes que hacer es poner tu chatbot frente a las partes interesadas. Para ello, implementará su chatbot como un punto final FastAPI y creará una interfaz de usuario Streamlit para interactuar con el punto final.

Before you get started, create two new folders called `chatbot_frontend/` and `tests/` in your project’s root directory. You’ll also need to add some additional files and folders to `chatbot_api/`:

```
./
│
├── chatbot_api/
│   │
│   ├── src/
│   │   │
│   │   ├── agents/
│   │   │   └── hospital_rag_agent.py
│   │   │
│   │   ├── chains/
│   │   │   ├── hospital_cypher_chain.py
│   │   │   └── hospital_review_chain.py
│   │   │
│   │   ├── models/
│   │   │   └── hospital_rag_query.py
│   │   │
│   │   ├── tools/
│   │   │   └── wait_times.py
│   │   │
│   │   ├── utils/
│   │   │   └── async_utils.py
│   │   │
│   │   ├── entrypoint.sh
│   │   └── main.py
│   │
│   ├── Dockerfile
│   └── pyproject.toml
│
├── chatbot_frontend/
│   │
│   ├── src/
│   │   ├── entrypoint.sh
│   │   └── main.py
│   │
│   ├── Dockerfile
│   └── pyproject.toml
│
├── hospital_neo4j_etl/
│   │
│   ├── src/
│   │   ├── entrypoint.sh
│   │   └── hospital_bulk_csv_write.py
│   │
│   ├── Dockerfile
│   └── pyproject.toml
│
├── tests/
│   ├── async_agent_requests.py
│   └── sync_agent_requests.py
│
├── .env
└── docker-compose.yml
```

Necesitas los nuevos archivos en `chatbot_api` para crear tu aplicación FastAPI, y `tests/` tiene dos scripts para demostrar el poder de hacer solicitudes asíncronas a tu agente. Por último, `chatbot_frontend/` tiene el código para la interfaz de usuario Streamlit que interactuará con su chatbot. Comenzará creando una aplicación FastAPI para servir a su agente.

### Sirve al agente con FastAPI

[FastAPI](https://realpython.com/fastapi-python-web-apis/) es un marco web moderno y de alto rendimiento para crear API con Python basada en sugerencias de tipo estándar. Viene con muchas características geniales, como la velocidad de desarrollo, la velocidad de tiempo de ejecución y un gran soporte de la comunidad, lo que lo convierte en una gran opción para servir a su agente de chatbot.

Servirá a su agente a través de una solicitud [POST](https://en.wikipedia.org/wiki/POST_(HTTP)), por lo que el primer paso es definir qué datos espera obtener en el cuerpo de la solicitud y qué datos devuelve la solicitud. FastAPI hace esto con [Pydantic](https://docs.pydantic.dev/latest/):

Python `chatbot_api/src/models/hospital_rag_query.py`

```python
from pydantic import BaseModel

class HospitalQueryInput(BaseModel):
    text: str

class HospitalQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]
```

In this script, you define Pydantic models `HospitalQueryInput` and `HospitalQueryOutput`. `HospitalQueryInput` is used to verify that the POST request body includes a `text` field, representing the query your chatbot responds to. `HospitalQueryOutput` verifies the response body sent back to your user includes `input`, `output`, and `intermediate_step` fields.

En este script, se definen los modelos Pydantic `HospitalQueryInput` y `HospitalQueryOutput`. `HospitalQueryInput` se utiliza para verificar que el cuerpo de la solicitud POST incluye un campo de `texto`, que representa la consulta a la que responde su chatbot. `HospitalQueryOutput` verifica que el cuerpo de la respuesta enviado de vuelta a su usuario incluye campos de `entrada`, `salida` y `inermediate_step`.

Una gran característica de FastAPI son sus capacidades de servicio [asíncrona](https://fastapi.tiangolo.com/async/). Debido a que su agente llama a los modelos OpenAI alojados en un servidor externo, siempre habrá latencia mientras su agente espera una respuesta. Esta es una oportunidad perfecta para que utilices la programación asíncrona.

En lugar de esperar a que OpenAI responda a cada una de las solicitudes de su agente, puede hacer que su agente haga varias solicitudes seguidas y almacene las respuestas a medida que se reciben. Esto le ahorrará mucho tiempo si tiene varias consultas a las que necesita que su agente responda.

Como se discutió anteriormente, a veces puede haber problemas de conexión intermitentes con Neo4j que generalmente se resuelven estableciendo una nueva conexión. Debido a esto, querrás implementar una lógica de reintento que funcione para funciones asíncronas:

Python `chatbot_api/src/utils/async_utils.py`

```python
import asyncio

def async_retry(max_retries: int=3, delay: int=1):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"Attempt {attempt} failed: {str(e)}")
                    await asyncio.sleep(delay)

            raise ValueError(f"Failed after {max_retries} attempts")

        return wrapper

    return decorator
```

No te preocupes por los detalles de `@async_retry`. Todo lo que necesitas saber es que volverá a intentar una función asíncrona si falla. Verás dónde se usa esto a continuación.

La lógica de conducción de su API de chatbot está en `chatbot_api/src/main.py`:

Python `chatbot_api/src/main.py`

```python
from fastapi import FastAPI
from agents.hospital_rag_agent import hospital_rag_agent_executor
from models.hospital_rag_query import HospitalQueryInput, HospitalQueryOutput
from utils.async_utils import async_retry

app = FastAPI(
    title="Hospital Chatbot",
    description="Endpoints for a hospital system graph RAG chatbot",
)

@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """Retry the agent if a tool fails to run.

    This can help when there are intermittent connection issues
    to external APIs.
    """
    return await hospital_rag_agent_executor.ainvoke({"input": query})

@app.get("/")
async def get_status():
    return {"status": "running"}

@app.post("/hospital-rag-agent")
async def query_hospital_agent(query: HospitalQueryInput) -> HospitalQueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    return query_response
```

Usted importa `FastAPI`, el ejecutor de su agente, los modelos Pydantic que creó para la solicitud POST y `@async_retry`. A continuación, establece una instancia de un objeto `FastAPI` y define `invoke_agent_with_retry()`, una función que ejecuta su agente de forma asíncrona. El decorador `@async_retry` de arriba `invoke_agent_with_retry()` garantiza que la función se volverá a intentar diez veces con un retraso de un segundo antes de fallar.

Por último, usted define `query_hospital_agent()` que sirve solicitudes POST a su agente en **/hospital-rag-agent**. Esta función extrae el campo de `texto` del cuerpo de la solicitud, lo pasa al agente y devuelve la respuesta del agente al usuario.

Servirás esta API con Docker y querrás definir el siguiente archivo de punto de entrada para que se ejecute dentro del contenedor:

Shell `chatbot_api/src/entrypoint.sh`

```shell
#!/bin/bash

# Run any setup steps or pre-processing tasks here
echo "Starting hospital RAG FastAPI service..."

# Start the main application
uvicorn main:app --host 0.0.0.0 --port 8000
```

The command `uvicorn main:app --host 0.0.0.0 --port 8000` runs the FastAPI application at port 8000 on your machine. The driving `Dockerfile`for your FastAPI app looks like this:

El comando `uvicorn main:app --host 0.0.0.0 --port 8000` ejecuta la aplicación FastAPI en el puerto 8000 de su máquina. El `Dockerfile` de conducción para tu aplicación FastAPI se ve así:

Dockerfile `chatbot_api/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY ./src/ /app

COPY ./pyproject.toml /code/pyproject.toml
RUN pip install /code/.

EXPOSE 8000
CMD ["sh", "entrypoint.sh"]
```

Este `Dockerfile` le dice a su contenedor que use la [distribution](https://hub.docker.com/_/python), `python:3.11-slim`, copie el contenido de `chatbot_api/src/` en el directorio `/app` dentro del contenedor, instale las dependencias de `pyproject.toml` y ejecute `entrypoint.sh`.

Lo último que tendrás que hacer es actualizar el archivo `docker-compose.yml` para incluir tu contenedor FastAPI:

```yml
version: '3'

services:
  hospital_neo4j_etl:
    build:
      context: ./hospital_neo4j_etl
    env_file:
      - .env

  chatbot_api:
    build:
      context: ./chatbot_api
    env_file:
      - .env
    depends_on:
      - hospital_neo4j_etl
    ports:
      - "8000:8000"
```

Aquí añades el servicio chatbot_api que se deriva del archivo Docker en `./chatbot_api`. Depende de hospital_neo4j_etl y se ejecutará en el puerto 8000.

Para ejecutar la API, junto con el ETL que construyó anteriormente, abra un terminal y ejecute:

Shell

```shellsession
$ docker-compose up --build
```

Si todo funciona correctamente, verás una pantalla similar a la siguiente en http://localhost:8000/docs#/:

[![FastAPI Docs](https://files.realpython.com/media/Screenshot_2024-01-14_at_3.26.14_PM.46d6c97c9bfd.png)](https://files.realpython.com/media/Screenshot_2024-01-14_at_3.26.14_PM.46d6c97c9bfd.png)

FastAPI docs screen

Puede usar la página de documentos para probar el endpoint `hospital-rag-agent`, pero no podrá hacer solicitudes asíncronas aquí. Para ver cómo su punto final maneja las solicitudes asíncronas, puede probarlo con una biblioteca como  [httpx](https://www.python-httpx.org/).

> **Nota**: Debe [install httpx](https://pypi.org/project/httpx/) en su entorno virtual antes de ejecutar las siguientes pruebas.

Para ver cuánto tiempo le ahorran las solicitudes asíncronas, comience estableciendo un punto de referencia utilizando solicitudes sincrónicas. Crea el siguiente script:

Python `tests/sync_agent_requests.py`

```python
import time
import requests

CHATBOT_URL = "http://localhost:8000/hospital-rag-agent"

questions = [
   "What is the current wait time at Wallace-Hamilton hospital?",
   "Which hospital has the shortest wait time?",
   "At which hospitals are patients complaining about billing and insurance issues?",
   "What is the average duration in days for emergency visits?",
   "What are patients saying about the nursing staff at Castaneda-Hardy?",
   "What was the total billing amount charged to each payer for 2023?",
   "What is the average billing amount for Medicaid visits?",
   "How many patients has Dr. Ryan Brown treated?",
   "Which physician has the lowest average visit duration in days?",
   "How many visits are open and what is their average duration in days?",
   "Have any patients complained about noise?",
   "How much was billed for patient 789's stay?",
   "Which physician has billed the most to cigna?",
   "Which state had the largest percent increase in Medicaid visits from 2022 to 2023?",
]

request_bodies = [{"text": q} for q in questions]

start_time = time.perf_counter()
outputs = [requests.post(CHATBOT_URL, json=data) for data in request_bodies]
end_time = time.perf_counter()

print(f"Run time: {end_time - start_time} seconds")
```

En este script, importa las `requests` y el `time`, define la URL de su chatbot, crea una lista de preguntas y registra la cantidad de tiempo que se tarda en obtener una respuesta a todas las preguntas de la lista. Si abres un terminal y ejecutas `sync_agent_requests.py`, verás cuánto tiempo se tarda en responder a las 14 preguntas:

Shell

```shell
(venv) $ python tests/sync_agent_requests.py
Run time: 68.20339595794212 seconds
```

Puede obtener resultados ligeramente diferentes dependiendo de su velocidad de Internet y la disponibilidad del modelo de chat, pero puede ver que este script tardó alrededor de 68 segundos en ejecutarse. A continuación, obtendrás respuestas a las mismas preguntas de forma asíncrona:

Python `tests/async_agent_requests.py`

```python
import asyncio
import time
import httpx

CHATBOT_URL = "http://localhost:8000/hospital-rag-agent"

async def make_async_post(url, data):
    timeout = httpx.Timeout(timeout=120)
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, timeout=timeout)
        return response

async def make_bulk_requests(url, data):
    tasks = [make_async_post(url, payload) for payload in data]
    responses = await asyncio.gather(*tasks)
    outputs = [r.json()["output"] for r in responses]
    return outputs

questions = [
   "What is the current wait time at Wallace-Hamilton hospital?",
   "Which hospital has the shortest wait time?",
   "At which hospitals are patients complaining about billing and insurance issues?",
   "What is the average duration in days for emergency visits?",
   "What are patients saying about the nursing staff at Castaneda-Hardy?",
   "What was the total billing amount charged to each payer for 2023?",
   "What is the average billing amount for Medicaid visits?",
   "How many patients has Dr. Ryan Brown treated?",
   "Which physician has the lowest average visit duration in days?",
   "How many visits are open and what is their average duration in days?",
   "Have any patients complained about noise?",
   "How much was billed for patient 789's stay?",
   "Which physician has billed the most to cigna?",
   "Which state had the largest percent increase in Medicaid visits from 2022 to 2023?",
]

request_bodies = [{"text": q} for q in questions]

start_time = time.perf_counter()
outputs = asyncio.run(make_bulk_requests(CHATBOT_URL, request_bodies))
end_time = time.perf_counter()

print(f"Run time: {end_time - start_time} seconds")
```

En `async_agent_requests.py`, haces la misma solicitud que hiciste en `sync_agent_requests.py`, excepto que ahora usas `httpx` para hacer las solicitudes de forma asíncrona. Aquí están los resultados:

shell

```shell
(venv) $ python tests/async_agent_requests.py
Run time: 17.766680584056303 seconds
```

Una vez más, el tiempo exacto que tarda esto en ejecutarse puede variar para usted, pero puede ver que hacer 14 solicitudes de forma asíncrona fue aproximadamente cuatro veces más rápido. La implementación de su agente de forma asíncrona le permite escalar a un volumen de alta solicitud sin tener que aumentar sus demandas de infraestructura. Si bien siempre hay excepciones, servir puntos finales REST de forma asíncrona suele ser una buena idea cuando su código realiza solicitudes vinculadas a la red.

Con este punto final de FastAPI funcionando, has hecho que tu agente sea accesible para cualquier persona que pueda acceder al punto final. Esto es genial para integrar a tu agente en las interfaces de usuario del chatbot, que es lo que harás a continuación con Streamlit.

### Crear una interfaz de usuario de chat con Streamlit

Sus partes interesadas necesitan una forma de interactuar con su agente sin hacer solicitudes manuales de API. Para adaptarse a esto, creará una aplicación [Streamlit](https://streamlit.io/) que actúa como una interfaz entre sus partes interesadas y su API. Estas son las dependencias de la interfaz de usuario de Streamlit:

TOML `chatbot_frontend/pyproject.toml`

```toml
[project]
name = "chatbot_frontend"
version = "0.1"
dependencies = [
   "requests==2.31.0",
   "streamlit==1.29.0"
]

[project.optional-dependencies]
dev = ["black", "flake8"]
```

El código de conducción de su aplicación Streamlit está en `chatbot_frontend/src/main.py`:

Python `chatbot_frontend/src/main.py`

```python
import os
import requests
import streamlit as st

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8000/hospital-rag-agent")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agent designed to answer questions about the hospitals, patients,
        visits, physicians, and insurance payers in  a fake hospital system.
        The agent uses  retrieval-augment generation (RAG) over both
        structured and unstructured data that has been synthetically generated.
        """
    )

    st.header("Example Questions")
    st.markdown("- Which hospitals are in the hospital system?")
    st.markdown("- What is the current wait time at wallace-hamilton hospital?")
    st.markdown(
        "- At which hospitals are patients complaining about billing and "
        "insurance issues?"
    )
    st.markdown("- What is the average duration in days for closed emergency visits?")
    st.markdown(
        "- What are patients saying about the nursing staff at "
        "Castaneda-Hardy?"
    )
    st.markdown("- What was the total billing amount charged to each payer for 2023?")
    st.markdown("- What is the average billing amount for medicaid visits?")
    st.markdown("- Which physician has the lowest average visit duration in days?")
    st.markdown("- How much was billed for patient 789's stay?")
    st.markdown(
        "- Which state had the largest percent increase in medicaid visits "
        "from 2022 to 2023?"
    )
    st.markdown("- What is the average billing amount per day for Aetna patients?")
    st.markdown("- How many reviews have been written from patients in Florida?")
    st.markdown(
        "- For visits that are not missing chief complaints, "
        "what percentage have reviews?"
    )
    st.markdown(
        "- What is the percentage of visits that have reviews for each hospital?"
    )
    st.markdown(
        "- Which physician has received the most reviews for this visits "
        "they've attended?"
    )
    st.markdown("- What is the ID for physician James Cooper?")
    st.markdown(
        "- List every review for visits treated by physician 270. Don't leave any out."
    )

st.title("Hospital System Chatbot")
st.info(
    "Ask me questions about patients, visits, insurance payers, hospitals, "
    "physicians, reviews, and wait times!"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        response = requests.post(CHATBOT_URL, json=data)

        if response.status_code == 200:
            output_text = response.json()["output"]
            explanation = response.json()["intermediate_steps"]

        else:
            output_text = """An error occurred while processing your message.
            Please try again or rephrase your message."""
            explanation = output_text

    st.chat_message("assistant").markdown(output_text)
    st.status("How was this generated", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            "explanation": explanation,
        }
    )
```

Aprender Streamlit no es el foco de este tutorial, por lo que no obtendrás una descripción detallada de este código. Sin embargo, aquí hay una descripción general de alto nivel de lo que hace esta interfaz de usuario:

- Todo el historial de chat se almacena y muestra cada vez que el usuario realiza una nueva consulta.
- La interfaz de usuario toma la entrada del usuario y realiza una solicitud POST sincrónica al punto final del agente.
- La respuesta más reciente del agente se muestra en la parte inferior del chat y se adjunta al historial del chat.
- Una explicación de cómo el agente generó la respuesta que proporcionó al usuario. Esto es genial para fines de auditoría porque puede ver si el agente llamó a la herramienta correcta, y puede comprobar si la herramienta funcionó correctamente.

Como lo has hecho, crearás un archivo de punto de entrada para ejecutar la interfaz de usuario:

Shell `chatbot_frontend/src/entrypoint.sh`

```shell
#!/bin/bash

# Run any setup steps or pre-processing tasks here
echo "Starting hospital chatbot frontend..."

# Run the ETL script
streamlit run main.py
```

Y, por último, el archivo Docker para crear una imagen para la interfaz de usuario:

Dockerfile `chatbot_frontend/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY ./src/ /app

COPY ./pyproject.toml /code/pyproject.toml
RUN pip install /code/.

CMD ["sh", "entrypoint.sh"]
```

Este `Dockerfile` es idéntico a los anteriores que has creado. Con eso, estás listo para ejecutar toda tu aplicación de chatbot end-to-end.

### Orquestación del proyecto con Docker Compose

En este punto, has escrito todo el código necesario para ejecutar tu chatbot. Este último paso es construir y ejecutar su proyecto con `docker-compose`. Antes de hacerlo, asegúrese de tener todos los siguientes archivos y carpetas en el directorio de su proyecto:

```
./
│
├── chatbot_api/
│   │
│   │
│   ├── src/
│   │   │
│   │   ├── agents/
│   │   │   └── hospital_rag_agent.py
│   │   │
│   │   ├── chains/
│   │   │   │
│   │   │   ├── hospital_cypher_chain.py
│   │   │   └── hospital_review_chain.py
│   │   │
│   │   ├── models/
│   │   │   └── hospital_rag_query.py
│   │   │
│   │   ├── tools/
│   │   │   └── wait_times.py
│   │   │
│   │   ├── utils/
│   │   │   └── async_utils.py
│   │   │
│   │   ├── entrypoint.sh
│   │   └── main.py
│   │
│   ├── Dockerfile
│   └── pyproject.toml
│
├── chatbot_frontend/
│   │
│   ├── src/
│   │   ├── entrypoint.sh
│   │   └── main.py
│   │
│   ├── Dockerfile
│   └── pyproject.toml
│
├── hospital_neo4j_etl/
│   │
│   ├── src/
│   │   ├── entrypoint.sh
│   │   └── hospital_bulk_csv_write.py
│   │
│   ├── Dockerfile
│   └── pyproject.toml
│
├── tests/
│   ├── async_agent_requests.py
│   └── sync_agent_requests.py
│
├── .env
└── docker-compose.yml
```

Su archivo `.env` debería tener las siguientes variables de entorno. La mayoría de ellos los creaste anteriormente en este tutorial, pero también tendrás que añadir uno nuevo para `CHATBOT_URL` para que tu aplicación Streamlit pueda encontrar tu API:

`.env`

```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>

NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_USERNAME>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>

HOSPITALS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/hospitals.csv
PAYERS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/payers.csv
PHYSICIANS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/physicians.csv
PATIENTS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/patients.csv
VISITS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/visits.csv
REVIEWS_CSV_PATH=https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/reviews.csv

HOSPITAL_AGENT_MODEL=gpt-3.5-turbo-1106
HOSPITAL_CYPHER_MODEL=gpt-3.5-turbo-1106
HOSPITAL_QA_MODEL=gpt-3.5-turbo-0125

CHATBOT_URL=http://host.docker.internal:8000/hospital-rag-agent
```

Para completar tu archivo `docker-compose.yml`, tendrás que añadir un servicio `chatbot_frontend`. Tu archivo final `docker-compose.yml` debería tener este aspecto:

Yaml `docker-compose.yml`

```yml
version: '3'

services:
  hospital_neo4j_etl:
    build:
      context: ./hospital_neo4j_etl
    env_file:
      - .env

  chatbot_api:
    build:
      context: ./chatbot_api
    env_file:
      - .env
    depends_on:
      - hospital_neo4j_etl
    ports:
      - "8000:8000"

  chatbot_frontend:
    build:
      context: ./chatbot_frontend
    env_file:
      - .env
    depends_on:
      - chatbot_api
    ports:
      - "8501:8501"
```

Por último, abre una terminal y ejecuta:

```shell
$ docker-compose up --build
```

Una vez que todo se construya y se ejecute, puedes acceder a la interfaz de usuario en http://localhost:8501/ y comenzar a chatear con tu chatbot:

[![Chatbot de trabajo](https://files.realpython.com/media/Screenshot_2024-01-14_at_8.53.00_PM.4d035582b8ce.png)](https://files.realpython.com/media/Screenshot_2024-01-14_at_8.53.00_PM.4d035582b8ce.png)

Chatbot del sistema hospitalario que funciona

Has creado un chatbot de extremo a extremo del sistema hospitalario en pleno funcionamiento. Tómese un tiempo para hacerle preguntas, ver el tipo de preguntas que es bueno para responder, averiguar dónde falla y pensar en cómo podría mejorarlo con mejores indicaciones o datos. Puede comenzar asegurándose de que las preguntas de ejemplo en la barra lateral se respondan correctamente.

## Conclusión

¡Enhorabuena por completar este tutorial en profundidad!

Has diseñado, construido y servido con éxito un chatbot de RAG LangChain que responde a preguntas sobre un sistema hospitalario falso. Ciertamente, hay muchas maneras de mejorar el chatbot que creaste en este tutorial, pero ahora tienes una sólida comprensión de cómo integrar LangChain con tus propios datos, lo que te da la libertad creativa para construir todo tipo de chatbots personalizados.

**En este tutorial, has aprendido a:**

- Usa **LangChain** para crear **chatbots** personalizados.
- Crea un chatbot para un sistema hospitalario falso alineándose con **los requisitos del negocio** y **aprovechando los datos disponibles**.
- Considere la implementación de **bases de datos de gráficos** en el diseño de su chatbot.
- Configure una instancia **Neo4j AuraDB** para su proyecto.
- Desarrolla un chatbot **RAG** capaz de recuperar datos tanto **estructurados**como **no estructurados** de Neo4j.
- Implementa tu chatbot usando **FastAPI** y **Streamlit**.

Puede encontrar el código fuente completo y los datos de este proyecto, en los materiales de apoyo, que puede descargar utilizando el siguiente enlace:

**Obtenga su código:** [Haga clic aquí para descargar el código fuente gratuito](https://realpython.com/bonus/build-llm-rag-chatbot-with-langchain-code/) de su chatbot LangChain.

Recibe un breve y dulce **truco** de **Python** en tu bandeja de entrada cada dos días. Nunca hay spam. Cancelar la suscripción en cualquier momento. Comisariado por el equipo de Real Python.
