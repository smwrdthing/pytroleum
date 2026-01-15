import numpy as np
import matplotlib.pyplot as plt
from pytroleum.sdyna.controllers import PropIntDiff


def simulate_pid(K, Ki, Kd, filter_val, T=20, dt=0.01):
    """Симуляция PID и возврат probe и времени"""
    pid = PropIntDiff(
        gain_coeff=K,
        integral_coeff=Ki,
        derivative_coeff=Kd,
        filter=filter_val,
        setpoint=1
    )

    probe = 0.0
    time = [0.0]
    probes = [probe]

    while time[-1] < T:
        pid.control(dt, probe)
        probe += pid.signal
        probes.append(probe)
        time.append(time[-1] + dt)

    return np.array(time), np.array(probes)


def test_pid():
    """Тест сравнения установившихся значений"""
    # Параметры для разных PID
    test_cases = [
        {"name": "PID1", "K": 0.01, "Ki": 0.01, "Kd": 0.95, "filter": 100},
        {"name": "PID2", "K": 0.02, "Ki": 0.01, "Kd": 0.95, "filter": 100},
        {"name": "PID3", "K": 0.01, "Ki": 0.02, "Kd": 0.95, "filter": 100},
    ]

    T = 20
    results = {}

    # Собираем результаты
    for case in test_cases:
        time, probes = simulate_pid(
            case["K"], case["Ki"], case["Kd"], case["filter"], T
        )
        results[case["name"]] = {
            "time": time,
            "probes": probes,
            "final": probes[-1]
        }

    # Проверяем равенство y1(T)=y2(T)=y3(T)
    final_values = [data["final"] for data in results.values()]

    # Проверка с допуском 0.001
    assert np.allclose(final_values, final_values[0], rtol=0.001), \
        f"Установившиеся значения различаются: {final_values}"

    # Сохраняем график
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data["time"], data["probes"],
                 label=f"{name}: {data['final']:.4f}")

    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Setpoint')
    plt.title(f'Probe от времени при разных коэффициентах (T={T})')
    plt.xlabel('time [-]')
    plt.ylabel('probe [-]')
    plt.legend()
    plt.grid(True)
    plt.savefig('pid_comparison.png')
    plt.show()
