import sys
import io
import logging
import time
from threading import Thread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time
try:
    import streamlit as st
except ModuleNotFoundError:
    sys.exit("Error: Streamlit is not available. Please install and run locally: `streamlit run app.py`.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import plotly.graph_objects as go

def plot_3d_surface_interactive(results_array, param_values, future_steps, param_to_vary):
    import streamlit as st
    import numpy as np

    X = np.arange(future_steps)
    Y = param_values
    X, Y = np.meshgrid(X, Y)
    Z = results_array

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        scene = dict(
            xaxis_title='Шаги времени',
            yaxis_title=param_to_vary.split('(')[0].strip(),
            zaxis_title='Общая численность'
        ),
        title=f"Интерактивный 3D-график: {param_to_vary.split('(')[0].strip()} / время / численность",
        autosize=True,
        margin=dict(l=40, r=40, b=40, t=40)
    )
    st.plotly_chart(fig, use_container_width=True)


#


def simulate_hybrid(
    N0_vec: list, T: int,
    fert_base: list, surv_base: list,
    K: float, r_fert: float, r_surv: float,
    delay_fert: list, delay_surv: list,
    migration_rates: list = None,
    env_effect: float = 0.0,
    stoch_intensity: float = 0.1,
    features: dict = None
) -> np.ndarray:
    # Значения по умолчанию, если не переданы
    if features is None:
        features = {
            "Плотностная зависимость рождаемости": True,
            "Плотностная зависимость выживаемости": True,
            "Задержки рождаемости": True,
            "Задержки выживаемости": True,
            "Миграция между группами": True,
            "Случайные колебания": True,
            "Влияние среды": True
        }

    n = len(N0_vec)
    N = np.array(N0_vec, dtype=float)
    history = [N.copy()]

    # Буфер для истории (для задержек)
    buffer_size = (max(delay_fert) if features["Задержки рождаемости"] else 0) + \
                  (max(delay_surv) if features["Задержки выживаемости"] else 0) + 1
    buffer = [N.copy()] * buffer_size

    # Миграционная матрица (по умолчанию нет миграции)
    if migration_rates is None or not features["Миграция между группами"]:
        migration_rates = [0.0] * n

    total_pop = np.sum(N)

    for t in range(T):
        # Текущее состояние
        N_new = np.zeros(n)
        total_pop = sum(buffer[-1])

        # Стохастический компонент
        noise = (np.random.normal(0, stoch_intensity * np.sqrt(buffer[-1] + 1))
                 if features["Случайные колебания"] else np.zeros(n))

        # Влияние среды
        env_factor = (1.0 + env_effect * np.sin(t * 0.1)
                      if features["Влияние среды"] else 1.0)

        # Рождаемость с плотностной зависимостью и задержкой
        for i in range(n):
            # Определяем популяцию для расчета - с задержкой или текущая
            delayed_pop = (buffer[-delay_fert[i]][i]
                           if features["Задержки рождаемости"]
                           else buffer[-1][i])

            # Плотностная зависимость рождаемости
            density_effect = (np.exp(-r_fert * (total_pop / K))
                              if features["Плотностная зависимость рождаемости"]
                              else 1.0)

            fertility = fert_base[i] * density_effect * env_factor
            N_new[0] += fertility * buffer[-1][i]

        # Выживаемость с плотностной зависимостью и задержкой
        for i in range(1, n):
            # Определяем популяцию для расчета - с задержкой или текущая
            delayed_pop = (buffer[-delay_surv[i - 1]][i - 1]
                           if features["Задержки выживаемости"]
                           else buffer[-1][i - 1])

            # Плотностная зависимость выживаемости
            density_effect = (np.exp(-r_surv * (delayed_pop / (K / n)))
                              if features["Плотностная зависимость выживаемости"]
                              else 1.0)

            survival = surv_base[i - 1] * density_effect * env_factor
            N_new[i] += survival * buffer[-1][i - 1]

        # Миграция между возрастными классами
        if features["Миграция между группами"]:
            migration = np.zeros(n)
            for i in range(n):
                outflow = buffer[-1][i] * migration_rates[i]
                migration[i] -= outflow
                # Распределяем оттоки равномерно по другим классам
                for j in range(n):
                    if i != j:
                        migration[j] += outflow / (n - 1)
            N_new += migration

        # Добавляем шум и ограничиваем минимальную популяцию
        N_new = np.clip(N_new + noise, 0, None)

        # Обновляем буфер и историю
        buffer.append(N_new)
        if len(buffer) > buffer_size:
            buffer.pop(0)

        history.append(N_new.copy())

    return np.array(history)
def simulate_logistic(N0: float, r: float, K: float, T: int) -> np.ndarray:
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] + r * Ns[-1] * (1 - Ns[-1] / K))
    return np.array(Ns)


def generate_hybrid_heatmap(base_params, param_name, param_range, steps, future_steps):
    param_values = np.linspace(param_range[0], param_range[1], steps)
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, val in enumerate(param_values):
        # Обновляем параметр
        modified_params = base_params.copy()
        modified_params[param_name] = val

        # Запускаем симуляцию
        history = simulate_hybrid(**modified_params)
        total_pop = history.sum(axis=1)
        results.append(total_pop[-future_steps:])  # Берем только последние future_steps шагов

        # Обновляем прогресс
        progress_bar.progress((i + 1) / steps)
        status_text.text(f"Прогресс: {i + 1}/{steps} ({param_name} = {val:.2f})")

    # Строим тепловую карту
    results_array = np.array(results)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(results_array, aspect='auto', cmap='viridis')

    # Настройки осей
    ax.set_xlabel("Шаги времени")
    ax.set_ylabel(param_name)
    ax.set_title(f"Тепловая карта популяции при изменении {param_name}")

    # Подписи значений параметра
    yticks = np.linspace(0, len(param_values) - 1, min(5, len(param_values)))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{param_values[int(i)]:.2f}" for i in yticks])

    # Цветовая шкала
    cbar = fig.colorbar(im)
    cbar.set_label("Общая численность популяции")

    st.pyplot(fig)
    return results_array
def simulate_ricker(N0: float, r: float, K: float, T: int) -> np.ndarray:
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] * np.exp(r * (1 - Ns[-1] / K)))
    return np.array(Ns)

def simulate_leslie(N0_vec: list, fertility: list, survival: list, T: int) -> np.ndarray:
    n = len(N0_vec)
    N = np.array(N0_vec, dtype=float)
    history = [N.copy()]
    L = np.zeros((n, n))
    L[0, :] = fertility
    for i in range(1, n):
        L[i, i-1] = survival[i-1]
    for _ in range(T):
        N = L.dot(N)
        history.append(N.copy())
    return np.array(history)

def simulate_delay(N0: float, r: float, K: float, T: int, tau: int) -> np.ndarray:
    # Создаем историю с начальными значениями
    Ns = [N0] * (tau + 1)
    # Симулируем T шагов
    for t in range(tau, T + tau):
        N_next = Ns[t] * np.exp(r * (1 - Ns[t - tau] / K))
        Ns.append(N_next)
    return np.array(Ns[:T + 1])  # Возвращаем только T+1 точек

def simulate_stochastic(base_sim, *args, sigma: float = 0.1, repeats: int = 100) -> np.ndarray:
    runs = []
    progress = st.progress(0)
    for i in range(repeats):
        traj = base_sim(*args)
        noise = np.random.normal(0, sigma, size=traj.shape)
        runs.append(np.clip(traj + noise, 0, None))
        progress.progress((i + 1) / repeats)
    return np.array(runs)
def send_to_gpt(typem,str):
        import g4f
        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_4,
            messages=[{"role": "user", "content": f"Воспринимай график как данные точек.Проанализируй график или возможно несколько графиков популяционной модели.Ничего не проси уточнить. Это не чат ты пишешь 1 раз и всё.Обязательно форматируй текст по MakrDown. будто ты научный сотрудник. Тип модели:{typem} вот результат симуляции: {str}"}],
            #stream=True
        )  # alternative model setting
        return response
def export_csv(data, filename,typem,str):
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать данные CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv"
    )
    with st.expander(label="Анализ модели",expanded=True):
        container = st.container(border=True)
        container.write('Анализ модели:')
        container.write(send_to_gpt(typem,str))



st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")
st.title("🌱 Симулятор Популяционной Динамики")

model_info = {
    "Гибридная модель": "Интегративная модель с возрастной структурой, плотностной зависимостью, задержками, стохастичностью и пространственной структурой.",
    "Логистический рост": "Классическая логистическая карта с предельной численностью K.",
    "Модель Рикера": "Экспоненциальный рост с зависимостью от плотности (Рикер).",
    "Модель Лесли": "Возрастная структура модели через матрицу Лесли.",
    "Модель с задержкой": "Популяция зависит от прошлого состояния (задержка τ).",
    "Стохастическая симуляция": "Добавляет гауссов шум к нескольким запускам.",
}
st.sidebar.info("Выберите модель и установите параметры ниже.")

model = st.sidebar.selectbox("Выберите модель:", list(model_info.keys()))
st.sidebar.caption(model_info[model])

st.sidebar.markdown("### Общие параметры")
T = st.sidebar.number_input("Шаги времени (T)", min_value=1, max_value=500, value=100)

common = {}
if model != "Модель Лесли":
    common['N0'] = st.sidebar.number_input("Начальная популяция N0", min_value=0.0, value=10.0)
    common['r'] = st.sidebar.number_input("Темп роста r", min_value=0.0, value=0.1)
    common['K'] = st.sidebar.number_input("Емкость K", min_value=1.0, value=100.0)

if model == "Модель с задержкой":
    tau_values = st.sidebar.multiselect(
        "Значения задержки (τ)",
        options=list(range(1, 11)),
        default=[1, 2]
    )
elif model == "Гибридная модель":
    n = st.sidebar.number_input(
        "Число возрастных групп",
        min_value=1 ,
        max_value=10,
        value=3,
        help="Например: 3 группы = молодые/взрослые/старые"
    )
    st.sidebar.markdown("### Параметры модели")

    model_features = {
            "Плотностная зависимость рождаемости": True,
            "Плотностная зависимость выживаемости": True,
            "Задержки рождаемости": True,
            "Задержки выживаемости": True,
            "Миграция между группами": True,
            "Случайные колебания": True,
            "Влияние среды": True
    }

    for feature, default in model_features.items():
        model_features[feature] = st.sidebar.toggle(feature, value=default)
    with st.sidebar.expander("Начальная численность"):
        st.markdown("""
            <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
            Сколько особей в каждой группе в начале моделирования
            </div>
            """, unsafe_allow_html=True)
        N0_vec = [
            st.number_input(
                f"🔢 Группа {i + 1} (молодые)" if i == 0 else
                f"🔢 Группа {i + 1} (взрослые)" if i == 1 else
                f"🔢 Группа {i + 1} (старые)",
                min_value=0.0,
                value=10.0
            ) for i in range(n)
        ]

    if model_features["Плотностная зависимость рождаемости"]:
        with st.sidebar.expander("Рождаемость"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Сколько потомков производит одна особь из этой группы
                </div>
                """, unsafe_allow_html=True)
            fert_base = [
                st.number_input(
                    f"👶 Детей на 1 особь группы {i + 1}",
                    min_value=0.0,
                    value=0.5,
                    help=f"Например: 0.5 = 1 особь производит 0.5 потомков в среднем"
                ) for i in range(n)
            ]
    else:
        # Если параметр неактивен, можно либо не показывать ничего, либо задать дефолт
        fert_base = [0.0] * n  # или любое дефолтное безопасное значение

    if model_features["Плотностная зависимость выживаемости"]:
        with st.sidebar.expander("Выживаемость"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Вероятность перехода в следующую возрастную группу
                </div>
                """, unsafe_allow_html=True)
            surv_base = [
                st.number_input(
                    f"🔄 Группа {i + 1} → Группа {i + 2}",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    help=f"0.8 = 80% особей перейдут в следующую группу"
                ) for i in range(n - 1)
            ]
    else:
        surv_base = [0.0] * (n - 1)

    if model_features["Задержки рождаемости"]:
        with st.sidebar.expander("Задержка реакции рождаемости"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Через сколько шагов рождаемость реагирует на изменения
                </div>
                """, unsafe_allow_html=True)
            delay_fert = [
                st.number_input(
                    f"⏳ Группа {i + 1} (шагов)",
                    min_value=0,
                    max_value=5,
                    value=1,
                    help="0 = мгновенная реакция, 1 = реагирует через 1 шаг"
                ) for i in range(n)
            ]
    else:
        delay_fert = [0] * n

    if model_features["Задержки выживаемости"]:
        with st.sidebar.expander("Задержка реакции выживаемости"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Через сколько шагов выживаемость реагирует на изменения
                </div>
                """, unsafe_allow_html=True)
            delay_surv = [
                st.number_input(
                    f"⏳ Переход {i + 1}→{i + 2} (шагов)",
                    min_value=0,
                    max_value=5,
                    value=1
                ) for i in range(n - 1)
            ]
    else:
        delay_surv = [0] * (n - 1)

    if model_features["Миграция между группами"]:
        with st.sidebar.expander("Миграция между группами"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Какая доля особей переходит в другие группы каждый шаг
                </div>
                """, unsafe_allow_html=True)
            migration_rates = [
                st.number_input(
                    f"🔄 Группа {i + 1} (доля мигрантов)",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    help="0.1 = 10% особей уйдут в другие группы"
                ) for i in range(n)
            ]
    else:
        migration_rates = [0.0] * n

    st.sidebar.markdown("---")
    K = st.sidebar.number_input(
        "📊 Максимальная численность (K)",
        min_value=1.0,
        value=100.0,
        help="Предел, который среда может поддерживать"
    )
    r_fert = st.sidebar.number_input(
        "📉 Влияние плотности на рождаемость",
        min_value=0.0,
        value=0.1,
        help="Чем больше, тем сильнее падает рождаемость при росте популяции"
    )
    r_surv = st.sidebar.number_input(
        "📉 Влияние плотности на выживаемость",
        min_value=0.0,
        value=0.05,
        help="Чем больше, тем сильнее падает выживаемость при росте популяции"
    )
    if model_features["Влияние среды"]:
        env_effect = st.sidebar.slider(
            "🌡️ Влияние среды",
            min_value=-1.0,
            max_value=1.0,
            value=0.2,
            help="-1: кризис, 0: нейтрально, +1: благоприятные условия"
        )
    else:
        env_effect = 0.0

    if model_features["Случайные колебания"]:
        stoch_intensity = st.sidebar.slider(
            "🎲 Случайные колебания",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            help="0: нет случайности, 1: сильные случайные изменения"
        )
    else:
        stoch_intensity = 0.0

elif model == "Модель Лесли":
    n = st.sidebar.number_input("Число возрастных классов", min_value=2, max_value=10, value=3)
    with st.sidebar.expander("Коэффициенты рождаемости (f_i)"):
        fertility = [st.number_input(f"f_{i}", min_value=0.0, value=0.5) for i in range(n)]
    with st.sidebar.expander("Вероятности выживания (s_i)"):
        survival = [st.number_input(f"s_{i}", min_value=0.0, max_value=1.0, value=0.8) for i in range(n-1)]
    with st.sidebar.expander("Начальная популяция по возрастным классам"):
        N0_vec = [st.number_input(f"N0_{i}", min_value=0.0, value=10.0) for i in range(n)]

elif model == "Стохастическая симуляция":
    repeats = st.sidebar.number_input("Число повторений", min_value=1, max_value=200, value=50)
    sigma_values = st.sidebar.multiselect(
        "Значения шума (σ)",
        options=[0.0, 0.05, 0.1, 0.2, 0.5],
        default=[0.1]
    )
    base_model = st.sidebar.selectbox("Основная модель:", ["Логистический рост", "Модель Рикера"])
    base_sim = simulate_logistic if base_model == "Логистический рост" else simulate_ricker

else:
    configs_count = st.sidebar.number_input("Количество конфигураций", min_value=1, max_value=5, value=1)
    config_params = []
    for i in range(configs_count):
        st.sidebar.markdown(f"**Конфигурация #{i+1}**")
        N0_i = st.sidebar.number_input(f"N0 (начальная популяция) #{i+1}", min_value=0.0, value=10.0)
        r_i = st.sidebar.number_input(f"r (темп роста) #{i+1}", min_value=0.0, value=0.1)
        K_i = st.sidebar.number_input(f"K (емкость) #{i+1}", min_value=1.0, value=100.0)
        config_params.append((N0_i, r_i, K_i))
st.sidebar.markdown('---')
with st.sidebar.expander("Настройка тепловой карты", expanded=True):
    heatmap_enabled = st.checkbox("Активировать тепловую карту", value=True)
    if heatmap_enabled:
        with st.sidebar.expander("Настройки тепловой карты"):
            param_to_vary = st.selectbox(
                "Параметр для анализа",
                options=[
                    "Влияние среды (env_effect)",
                    "Случайные колебания (stoch_intensity)",
                    "Емкость среды (K)",
                    "Коэффициент рождаемости (r_fert)"
                ],
                index=0
            )

            # Определяем текущее значение параметра
            current_value = {
                "Влияние среды (env_effect)": env_effect,
                "Случайные колебания (stoch_intensity)": stoch_intensity,
                "Емкость среды (K)": K,
                "Коэффициент рождаемости (r_fert)": r_fert
            }[param_to_vary]

            # Настройки диапазона в зависимости от параметра
            param_config = {
                "Влияние среды (env_effect)": {
                    "min": -1.0, "max": 1.0, "step": 0.1,
                    "range": (max(-1.0, current_value - 0.5), min(1.0, current_value + 0.5))
                },
                "Случайные колебания (stoch_intensity)": {
                    "min": 0.0, "max": 1.0, "step": 0.05,
                    "range": (0.0, min(1.0, current_value * 2))
                },
                "Емкость среды (K)": {
                    "min": 10.0, "max": 1000.0, "step": 10.0,
                    "range": (max(10.0, current_value * 0.5), current_value * 2)
                },
                "Коэффициент рождаемости (r_fert)": {
                    "min": 0.0, "max": 1.0, "step": 0.05,
                    "range": (0.0, min(1.0, current_value * 2))
                }
            }[param_to_vary]

            param_range = st.slider(
                f"Диапазон изменения {param_to_vary.split('(')[0]}",
                min_value=param_config["min"],
                max_value=param_config["max"],
                value=param_config["range"],
                step=param_config["step"]
            )

            steps = st.number_input("Число шагов параметра", min_value=2, max_value=20, value=5)
            future_steps = st.number_input("Шаги прогнозирования", min_value=10, max_value=1000, value=30)


if st.sidebar.button("Симулировать"):
    with st.spinner("Симуляция..."):
        if model == "Логистический рост":
            if configs_count == 1:
                traj = simulate_logistic(config_params[0][0], config_params[0][1], config_params[0][2], T)
                df = pd.DataFrame(traj, columns=["Популяция"])
                st.subheader("Логистический рост")
                st.line_chart(df)
                export_csv(df, 'logistic_growth', 'Логистический рост',
                           f"Одна траектория: N0={config_params[0][0]}, r={config_params[0][1]}, K={config_params[0][2]}\nДанные:\n{traj}")
            else:
                all_trajs = {}
                config_descriptions = []
                for idx, (N0_i, r_i, K_i) in enumerate(config_params):
                    traj = simulate_logistic(N0_i, r_i, K_i, T)
                    all_trajs[f"Конфигурация #{idx + 1} (r={r_i}, K={K_i})"] = traj
                    config_descriptions.append(f"Конфигурация #{idx + 1}: N0={N0_i}, r={r_i}, K={K_i}")
                df = pd.DataFrame(all_trajs)
                st.subheader("Логистический рост - Несколько конфигураций")
                st.line_chart(df)
                export_csv(df, 'logistic_growth_multiple', 'Логистический рост',
                           f"Множественные траектории:\n{'\n'.join(config_descriptions)}\nДанные:\n{all_trajs}")
        elif model == "Гибридная модель":
            if(n == 1 and model_features['Задержки выживаемости']):
                st.warning("Для одной возрастной группы не может быть задержек выживаемости!\nПараметр был отключён.")
                model_features['Задержки выживаемости'] = False
            history = simulate_hybrid(
                N0_vec, T, fert_base, surv_base, K,
                r_fert, r_surv, delay_fert, delay_surv,
                migration_rates, env_effect, stoch_intensity,
                features=model_features  # Передаем настройки toggle
            )

            df = pd.DataFrame(history, columns=[f"Возраст {i}" for i in range(n)])
            with st.expander("Гибридная модель - Разные классы",expanded=True):
                # Визуализация результатов
                st.subheader("Гибридная модель - Динамика по возрастным классам")
                st.line_chart(df)
            with st.expander("Гибридная модель - Общая статистика",expanded=False):
                # Общая численность популяции
                total_pop = df.sum(axis=1)
                st.subheader("Гибридная модель - Общая численность популяции")
                st.line_chart(pd.DataFrame(total_pop, columns=["Общая численность"]))


            # Параметры для экспорта
            params_str = (f"Возрастные классы: {n}, K={K}, r_fert={r_fert}, r_surv={r_surv}, "
                        f"env_effect={env_effect}, stoch_intensity={stoch_intensity}\n"
                        f"fert_base={fert_base}, surv_base={surv_base}\n"
                        f"delay_fert={delay_fert}, delay_surv={delay_surv}\n"
                        f"migration_rates={migration_rates}")
            # Тепловая карта (если активирована)
            if heatmap_enabled:
                st.subheader("🔍 Тепловая карта динамики популяции")

                # Соответствие между названиями параметров и переменными
                param_mapping = {
                    "Влияние среды (env_effect)": "env_effect",
                    "Случайные колебания (stoch_intensity)": "stoch_intensity",
                    "Емкость среды (K)": "K",
                    "Коэффициент рождаемости (r_fert)": "r_fert"
                }

                base_params = {
                    "N0_vec": N0_vec,
                    "T": T + future_steps,
                    "fert_base": fert_base,
                    "surv_base": surv_base,
                    "K": K,
                    "r_fert": r_fert,
                    "r_surv": r_surv,
                    "delay_fert": delay_fert,
                    "delay_surv": delay_surv,
                    "migration_rates": migration_rates,
                    "env_effect": env_effect,
                    "stoch_intensity": stoch_intensity,
                    "features": model_features
                }

                # Генерация данных
                param_values = np.linspace(param_range[0], param_range[1], steps)
                results = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, val in enumerate(param_values):
                    modified_params = base_params.copy()
                    modified_params[param_mapping[param_to_vary]] = val

                    history = simulate_hybrid(**modified_params)
                    total_pop = history.sum(axis=1)
                    results.append(total_pop[-future_steps:])

                    progress_bar.progress((i + 1) / steps)
                    status_text.text(f"Прогресс: {i + 1}/{steps} ({param_to_vary} = {val:.2f})")

                # Визуализация
                results_array = np.array(results)

                fig, ax = plt.subplots(figsize=(12, 6))
                im = ax.imshow(results_array, aspect='auto', cmap='viridis',
                               extent=[0, future_steps, param_range[0], param_range[1]])
                # ------ 3D график ------
                st.markdown("#### 3D-график динамики (ось Z — численность)")
                plot_3d_surface_interactive(results_array, param_values, future_steps, param_to_vary)

                ax.set_xlabel("Шаги времени")
                ax.set_ylabel(param_to_vary.split('(')[0].strip())
                ax.set_title(f"Тепловая карта: изменение {param_to_vary.split('(')[0].strip()}")
                plt.colorbar(im, label="Общая численность популяции")

                # Добавляем линию для текущего значения
                current_val = {
                    "Влияние среды (env_effect)": env_effect,
                    "Случайные колебания (stoch_intensity)": stoch_intensity,
                    "Емкость среды (K)": K,
                    "Коэффициент рождаемости (r_fert)": r_fert
                }[param_to_vary]

                if param_range[0] <= current_val <= param_range[1]:
                    y_pos = (current_val - param_range[0]) / (param_range[1] - param_range[0]) * steps
                    ax.axhline(y=current_val, color='red', linestyle='--', linewidth=1)
                    ax.text(0, current_val, ' Текущее значение', color='red', va='center')

                st.pyplot(fig)

            export_csv(df, 'hybrid_model', 'Гибридная модель', params_str)


        elif model == "Модель Рикера":
            if configs_count == 1:
                traj = simulate_ricker(config_params[0][0], config_params[0][1], config_params[0][2], T)
                df = pd.DataFrame(traj, columns=["Популяция"])
                st.subheader("Модель Рикера")
                st.line_chart(df)
                export_csv(df, 'ricker_model', 'Модель Рикера',
                           f"Одна траектория: N0={config_params[0][0]}, r={config_params[0][1]}, K={config_params[0][2]}\nДанные:\n{traj}")
            else:
                all_trajs = {}
                config_descriptions = []
                for idx, (N0_i, r_i, K_i) in enumerate(config_params):
                    traj = simulate_ricker(N0_i, r_i, K_i, T)
                    all_trajs[f"Конфигурация #{idx + 1} (r={r_i}, K={K_i})"] = traj
                    config_descriptions.append(f"Конфигурация #{idx + 1}: N0={N0_i}, r={r_i}, K={K_i}")
                df = pd.DataFrame(all_trajs)
                st.subheader("Модель Рикера - Несколько конфигураций")
                st.line_chart(df)
                export_csv(df, 'ricker_model_multiple', 'Модель Рикера',
                           f"Множественные траектории:\n{'\n'.join(config_descriptions)}\nДанные:\n{all_trajs}")


        elif model == "Модель с задержкой":
            if not tau_values:
                st.warning("Выберите хотя бы одно значение τ")
            else:
                all_trajs = {}
                tau_descriptions = []
                for tau_i in tau_values:
                    traj = simulate_delay(common['N0'], common['r'], common['K'], T, tau_i)
                    all_trajs[f"τ = {tau_i}"] = traj
                    tau_descriptions.append(
                        f"Задержка τ={tau_i} при N0={common['N0']}, r={common['r']}, K={common['K']}")
                df = pd.DataFrame(all_trajs)
                st.subheader("Модель с задержкой - Разные τ")
                st.line_chart(df)
                export_csv(df, 'delay_model_multiple_tau', 'Модель с задержкой',
                           f"Траектории с разными задержками:\n{'\n'.join(tau_descriptions)}\nДанные:\n{all_trajs}")

        elif model == "Модель Лесли":
            history = simulate_leslie(N0_vec, fertility, survival, T)
            df = pd.DataFrame(history, columns=[f"Возраст {i}" for i in range(n)])
            st.subheader("Модель Лесли")
            st.line_chart(df)
            L = np.zeros((n, n))
            L[0, :] = fertility
            for i in range(1, n):
                L[i, i - 1] = survival[i - 1]
            lambda_val = np.max(np.real(np.linalg.eigvals(L)))
            st.write(f"Доминирующее собственное значение λ = {lambda_val:.3f}")
            export_csv(df, 'leslie_matrix','Модель Лесли',history)


        elif model == "Стохастическая симуляция":
            if not sigma_values:
                st.warning("Выберите хотя бы одно значение σ")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                all_means = {}
                sigma_descriptions = []
                for sigma in sigma_values:
                    results = simulate_stochastic(
                        base_sim,
                        common['N0'],
                        common['r'],
                        common['K'],
                        T,
                        sigma=sigma,
                        repeats=repeats
                    )
                    for i in range(repeats):
                        ax.plot(results[i], alpha=0.1, linewidth=0.8)
                    mean_traj = results.mean(axis=0)
                    ax.plot(mean_traj, linewidth=2, label=f'σ={sigma}')
                    all_means[f"σ={sigma}"] = mean_traj
                    sigma_descriptions.append(f"σ={sigma} (N0={common['N0']}, r={common['r']}, K={common['K']})")
                ax.set_title(f"Стохастическая симуляция ({repeats} траекторий на сигму)")
                ax.legend()
                st.pyplot(fig)
                means_df = pd.DataFrame(all_means)
                st.subheader("Средние траектории для разных уровней шума")
                st.line_chart(means_df)
                export_csv(means_df, 'stochastic_simulation_means', 'Стохастическая модель',
                           f"Стохастические траектории с параметрами:\n{'\n'.join(sigma_descriptions)}\n"
                           f"Средние значения:\n{all_means}\n"
                           f"Базовые параметры: N0={common['N0']}, r={common['r']}, K={common['K']}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Разработано Лией Ахметовой — v1.0")
