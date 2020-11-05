import empymod
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from scipy.constants import mu_0
plt.style.use('ggplot')


def loop_tem1d(time, L_square, depth, res, verb_flag=0):

    # === GET REQUIRED FREQUENCIES ===
    time, freq, ft, ftarg = empymod.utils.check_time(
        time=time,          # Required times
        signal=-1,          # Switch-on response
        ft='dlf',           # Use DLF
        ftarg={'dlf': 'key_201_CosSin_2012'},  # Short, fast filter; if you
        # need higher accuracy choose a longer filter.
        verb=verb_flag,
    )

    # === COMPUTE FREQUENCY-DOMAIN RESPONSE ===
    # We only define a few parameters here. You could extend this for any
    # parameter possible to provide to empymod.model.bipole.
    EM = empymod.model.bipole(
        # El. bipole source; half of one side.
        src=[L_square/2., L_square/2., 0, L_square/2., 0, 0],
        rec=[0, 0, 0, 0, 90],         # Receiver at the origin, vertical.
        depth=np.r_[0, depth],        # Depth-model, adding air-interface.
        res=np.r_[2e14, res],         # Provided resistivity model, adding air.
        # aniso=aniso,                # Here you could implement anisotropy...
        #                             # ...or any parameter accepted by bipole.
        freqtime=freq,                # Required frequencies.
        mrec=True,                    # It is an el. source, but a magn. rec.
        strength=8,                   # To account for 4 sides of square loop.
        # Approx. the finite dip. with 3 points.
        srcpts=100,
        verb=verb_flag,
        # htarg={'dlf': 'key_101_2009'},  # Short filter, so fast.
    )

    # Multiply the frequecny-domain result with
    # \mu for H->B, and i\omega for B->dB/dt.
    EM_b = EM*mu_0
    EM_db = EM*2j*np.pi*freq*mu_0

    # === Butterworth-type filter (implemented from simpegEM1D.Waveforms.py)===
    # Note: Here we just apply one filter. But it seems that square_tem1d can apply
    #       two filters, one before and one after the so-called front gate
    #       (which might be related to ``delay_rst``, I am not sure about that
    #       part.)
    cutofffreq = 4.5e10
    h = (1+1j*freq/cutofffreq)**-1
    h *= (1+1j*freq/3e5)**-1
    EM_b *= h
    EM_db *= h

    # === CONVERT TO TIME DOMAIN ===
    delay_rst = 1.8e-7
    EM_b, _ = empymod.model.tem(EM_b[:, None], np.array([1]),
                                freq, time+delay_rst, -1, ft, ftarg)
    EM_b = np.squeeze(EM_b)
    EM_db, _ = empymod.model.tem(EM_db[:, None], np.array([1]),
                                 freq, time+delay_rst, -1, ft, ftarg)
    EM_db = np.squeeze(EM_db)

    # === APPLY WAVEFORM ===
    return EM_b, EM_db


if __name__ == '__main__':
    gatem_results = np.abs(np.loadtxt('gatem_hf-df-Bz.dat'))
    times = gatem_results[:, 0]
    L_square = 1
    depth = [100, ]
    res = [1e2, 1e2]
    EM_b, EM_db = loop_tem1d(times, L_square, depth, res, verb_flag=0)

    plt.figure(figsize=(9, 5))
    ax1 = plt.subplot(121)
    plt.title('Model Result')

    # empymod
    plt.plot(times, np.abs(gatem_results[:, 1]), 'b', label="GATEM_Fwd1D")
    plt.plot(times, np.abs(EM_db), 'r--', label="empymod")

    # Plot settings
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel("Time(s)")
    plt.ylabel(r"$\mathrm{B}_\mathrm{z}$")
    plt.grid(which='both', c='w')
    plt.legend(title='Data', loc=1)

    ax2 = plt.subplot(122)
    plt.title('Difference')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    plt.plot(times, np.abs(np.abs(EM_db)-np.abs(gatem_results[:, 1]))/np.abs(gatem_results[:, 1])*1e2, 'm.',
             label=r"$|\Delta\,\mathrm{B}_\mathrm{z}|$")

    # Plot settings
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel("Time(s)")
    plt.ylabel(r"Relative Error (%)")
    plt.ylim(0, 5)
    plt.legend(title='Difference', loc=3)

    # Force minor ticks on logscale
    ax1.yaxis.set_minor_locator(LogLocator(subs='all', numticks=20))
    ax2.yaxis.set_minor_locator(LogLocator(subs='all', numticks=20))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax2.yaxis.set_minor_formatter(NullFormatter())

    plt.grid(which='both', c='w')

    # Finish off
    plt.tight_layout()
    plt.show()
