import os
import json
from datetime import datetime
from pathlib import Path

import boto3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Langfuse podpinam tak samo jak w aplikacji NaszGPT (naszgpt_langfuse / naszgpt_digitalocean):
# dekorator @observe + klient OpenAI opakowany przez langfuse
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from langfuse.openai import OpenAI
# Dodatek: błąd zwracany przez OpenAI przy złym kluczu - wziąłem z dokumentacji biblioteki openai (obsługa błędów)
from openai import AuthenticationError

# Model wczytuję tak jak w lekcji 24__in_search_of_the_best_model__pycaret_flow.ipynb
from pycaret.regression import load_model, predict_model

load_dotenv()

#
# USTAWIENIA
#
BUCKET_NAME = "jerry-77"
# Model zapisuje notebook 02_pipeline_trenowania.ipynb, zawsze pod tym samym kluczem "latest"
MODEL_KEY = "polmaraton/models/model_polmaraton_latest.pkl"
METRICS_KEY = "polmaraton/models/model_polmaraton_latest.json"
MODELS_PATH = Path("models")
LLM_MODEL = "gpt-4o-mini"

# Dodatek: sprawdzam zakresy, żeby nie liczyć prognozy dla absurdalnych wartości.
# W df.describe() w notebooku 02 wiek był od 17 do 89 lat, a czas na 5 km od ok. 15 do 64 minut,
# więc biorę zaokrąglone granice z małym zapasem
MIN_AGE = 16
MAX_AGE = 90
MIN_5KM = 14 * 60  # w sekundach
MAX_5KM = 65 * 60

FIELD_NAMES = {
    "plec": "płeć",
    "wiek": "wiek",
    "czas_5km": "czas na 5 km",
}


#
# CHMURA - DIGITAL OCEAN SPACES
#
# Klienta boto3 tworzę jak w lekcji 10__cloud_storage.ipynb,
# endpoint podaję jawnie, bo Spaces to nie AWS
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
    )


# Dodatek: model pobieram z Spaces tylko raz na uruchomienie aplikacji (st.cache_resource),
# dzięki temu po wytrenowaniu nowej wersji wystarczy zrestartować aplikację.
# Plik JSON z metrykami modelu to też mój dodatek.
@st.cache_resource
def get_model():
    MODELS_PATH.mkdir(exist_ok=True)
    s3 = get_s3_client()
    s3.download_file(BUCKET_NAME, MODEL_KEY, "models/model_polmaraton_latest.pkl")
    s3.download_file(BUCKET_NAME, METRICS_KEY, "models/model_polmaraton_latest.json")

    with open("models/model_polmaraton_latest.json", "r", encoding="utf-8") as f:
        metrics = json.loads(f.read())

    # load_model z PyCaret sam dokleja ".pkl", tak jak w lekcji 24
    model = load_model("models/model_polmaraton_latest", verbose=False)
    return model, metrics


#
# POMOCNICZE - CZAS
#
# Funkcja na bazie snippetu z 41__zadanie_domowe.ipynb
# Dodatek: obsługa formatu MM:SS
def convert_time_to_seconds(time):
    if pd.isnull(time) or time in ['DNS', 'DNF']:
        return None
    time = time.split(':')
    if len(time) == 3:
        return int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
    if len(time) == 2:
        return int(time[0]) * 60 + int(time[1])
    return None


# Dodatek: zamiana w drugą stronę, żeby wynik pokazać jako H:MM:SS
def convert_seconds_to_time(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"


#
# LLM - WYCIĄGANIE DANYCH
#
# Prompt i sposób wywołania (response_format json_object + json.loads) wzorowałem na
# lekcji 33__monitorowanie_modeli__langfuse.ipynb (parsowanie maili wakacyjnych)
# Dodatek: w prompcie podpowiedź z polską gramatyką (płeć) i przeliczanie roku urodzenia na wiek
EXTRACTION_PROMPT = """
You are awesome helper in parsing the data about a runner.
You will be provided with the user self-introduction in polish that contains the following data:

<plec> - the gender of the user. Return "K" for a woman and "M" for a man.
Polish grammar can help you (e.g. "biegałam" - woman, "biegałem" - man).
<wiek> - the age of the user in years. If the user gives the year of birth instead,
calculate the age using the current year provided at the end of this message.
<czas_5km> - the time in which the user runs 5 kilometers. Convert it to the format "HH:MM:SS",
e.g. "25 minut" -> "00:25:00", "24:30" -> "00:24:30", "pół godziny" -> "00:30:00".

Return the value as a dictionary with the following keys:
- plec - as a string "K" or "M" or null if not provided
- wiek - as an integer or null if not provided
- czas_5km - as a string in format "HH:MM:SS" or null if not provided

Do not guess. If the information is not provided, return null for that key.
Return a valid dictionary, nothing else.
I will parse the result with json.loads() function in python so please make sure the result is valid.

Here is an example of the user content:
```
Cześć, mam na imię Ania, mam 32 lata i 5 km biegam w 27 minut.
```

In this case the dictionary should look like this:
{
"plec": "K",
"wiek": 32,
"czas_5km": "00:27:00"
}
"""


# Klient OpenAI biorę z session_state - tak jak w module 7
# (07__audio_notes_v3__protecting_openai_api_key.py)
def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])


@observe()
def get_data_from_intro(intro, model=LLM_MODEL):
    messages = [
        {
            "role": "system",
            "content": EXTRACTION_PROMPT + f"\nThe current year is {datetime.now().year}.",
        },
        {
            "role": "user",
            "content": f"```{intro}```",
        },
    ]
    openai_client = get_openai_client()
    chat_completion = openai_client.chat.completions.create(
        response_format={"type": "json_object"},
        messages=messages,
        model=model,
        name="get_data_from_intro",
    )
    resp = chat_completion.choices[0].message.content
    try:
        output = json.loads(resp)
    except:
        output = {"error": resp}

    # Metryka skuteczności LLM'a w Langfuse: jaka część potrzebnych pól została odczytana
    # Dodatek: w lekcji 33 score dodawałem przez langfuse.score(trace_id=...),
    # tu używam skrótu z dekoratora - score_current_trace (z dokumentacji langfuse)
    found = []
    for key in FIELD_NAMES:
        if output.get(key) is not None:
            found.append(key)
    langfuse_context.score_current_trace(
        name="kompletnosc-danych",
        value=len(found) / len(FIELD_NAMES),
        comment=f"Odczytane pola: {found}",
    )

    # Dodatek: id trace'a zapamiętuję, żeby później dopiąć do niego ocenę użytkownika
    output["trace_id"] = langfuse_context.get_current_trace_id()
    return output


# Sprawdzam, czy mam wszystko, czego potrzebuje model (wymaganie z zadania)
# Dodatek: sprawdzam też, czy wartości mają sens (zakresy wieku i czasu)
def validate_data(data):
    missing = []
    for key in FIELD_NAMES:
        if data.get(key) is None:
            missing.append(FIELD_NAMES[key])

    errors = []
    if data.get("plec") is not None and data["plec"] not in ["K", "M"]:
        errors.append("Nie udało się rozpoznać płci (oczekuję kobieta / mężczyzna).")

    if data.get("wiek") is not None:
        if data["wiek"] < MIN_AGE or data["wiek"] > MAX_AGE:
            errors.append(f"Wiek musi być w zakresie {MIN_AGE}-{MAX_AGE} lat.")

    if data.get("czas_5km") is not None:
        seconds = convert_time_to_seconds(data["czas_5km"])
        if seconds is None:
            errors.append("Nie udało się odczytać czasu na 5 km.")
        elif seconds < MIN_5KM or seconds > MAX_5KM:
            errors.append("Czas na 5 km musi być w zakresie od 14 do 65 minut.")
        else:
            data["czas_5km_s"] = seconds

    return missing, errors


def predict_halfmarathon_time(model, data):
    # Nazwy kolumn muszą być takie same jak w notebooku 02_pipeline_trenowania.ipynb
    input_data = {
        "plec": data["plec"],
        "wiek": data["wiek"],
        "czas_5km": data["czas_5km_s"],
    }
    # predict_model i kolumna prediction_label - tak jak w notebooku 02 (test zapisanego modelu)
    prediction_df = predict_model(model, data=pd.DataFrame([input_data]), verbose=False)
    return prediction_df["prediction_label"].iloc[0]


# Dodatek: ocena użytkownika (Tak / Nie) jako druga metryka w Langfuse
def send_user_feedback(trace_id, value):
    # Ocenę użytkownika zapisuję w Langfuse tak jak langfuse.score w lekcji 33
    Langfuse().score(
        trace_id=trace_id,
        name="ocena-uzytkownika",
        value=value,
        comment="Czy dane zostały poprawnie odczytane",
    )
    st.session_state["feedback_sent"] = True


#
# MAIN PROGRAM
#
st.set_page_config(page_title="Kalkulator półmaratonu", page_icon="🏃")
st.title("🏃 Ile zajmie Ci półmaraton?")
st.markdown(
    "Przedstaw się w kilku zdaniach. Napisz, **czy jesteś kobietą czy mężczyzną**, "
    "**ile masz lat** i **w jakim czasie biegasz 5 km**. "
    "Na tej podstawie oszacuję Twój czas na półmaratonie (21,0975 km)."
)

# Dodatek: zabezpieczenie klucza OpenAI - wziąłem je z modułu 7
# (07__audio_notes_v3__protecting_openai_api_key.py i 25__app.py z wdrażania aplikacji Notatki).
# Jeśli nie ma klucza w zmiennych środowiskowych, pokazuję pole, w którym można wpisać swój

# Dodatek: gdy OpenAI odrzuci klucz, pokazuję krótki komunikat zamiast długiego błędu technicznego
if st.session_state.get("bad_api_key"):
    st.error("Nieprawidłowy klucz API OpenAI. Sprawdź go i wpisz ponownie.")

if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in os.environ:
        st.session_state["openai_api_key"] = os.environ["OPENAI_API_KEY"]

    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            # nowy klucz wpisany - chowam komunikat o złym kluczu
            st.session_state["bad_api_key"] = False
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

intro = st.text_area(
    "Kilka słów o Tobie",
    placeholder="Np. Hej, tu Jerzy z Wrocławia. Biegam od kilku lat, mam 50 lat, a 5 km zajmuje mi około 27 minut.",
    height=150,
)

if st.button("Oszacuj czas", type="primary"):
    if not intro.strip():
        st.warning("Najpierw napisz coś o sobie :)")
    else:
        with st.spinner("Analizuję Twoją wiadomość..."):
            # Dodatek: przy złym kluczu czyszczę go i wracam do pola "Klucz API"
            try:
                st.session_state["data"] = get_data_from_intro(intro)
                st.session_state["feedback_sent"] = False
                st.session_state["bad_api_key"] = False
            except AuthenticationError:
                st.session_state["openai_api_key"] = None
                st.session_state["data"] = None
                st.session_state["bad_api_key"] = True
                st.rerun()

data = st.session_state.get("data")
if data:
    if "error" in data:
        st.error("Nie udało się przetworzyć odpowiedzi modelu językowego. Spróbuj ponownie.")
        st.stop()

    missing, errors = validate_data(data)

    with st.expander("Co udało mi się odczytać"):
        st.json({
            "plec": data.get("plec"),
            "wiek": data.get("wiek"),
            "czas_5km": data.get("czas_5km"),
        })

    if missing:
        st.warning(
            "Brakuje mi danych: **" + ", ".join(missing) + "**. "
            "Uzupełnij proszę swoją wiadomość i kliknij jeszcze raz."
        )
    elif errors:
        for error in errors:
            st.error(error)
    else:
        model, metrics = get_model()
        prediction = predict_halfmarathon_time(model, data)
        # Dodatek: oprócz czasu pokazuję średnie tempo na km
        pace = prediction / 21.0975

        c0, c1 = st.columns(2)
        with c0:
            st.metric("Szacowany czas półmaratonu", convert_seconds_to_time(prediction))
        with c1:
            st.metric("Średnie tempo", f"{int(pace // 60)}:{int(pace % 60):02d} min/km")

        # Dodatek: informacja, o ile średnio myli się model (MAE z pliku JSON)
        st.caption(
            f"Model myli się średnio o około {metrics['MAE'] / 60:.0f} min "
            f"(MAE na danych z półmaratonu wrocławskiego 2023-2024)."
        )

    # Dodatek: ocena użytkownika trafia do Langfuse jako druga metryka skuteczności LLM'a
    if data.get("trace_id") and not st.session_state.get("feedback_sent"):
        st.write("Czy poprawnie odczytałem Twoje dane?")
        c0, c1, _ = st.columns([1, 1, 3])
        with c0:
            st.button("✅ Tak", on_click=send_user_feedback, args=(data["trace_id"], 1))
        with c1:
            st.button("❌ Nie", on_click=send_user_feedback, args=(data["trace_id"], 0))
    elif st.session_state.get("feedback_sent"):
        st.caption("Dzięki za ocenę!")
