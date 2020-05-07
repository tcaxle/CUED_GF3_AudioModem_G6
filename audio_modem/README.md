# Audio Modem

> *Idealised System Model*
>
> This section describes a simple idealized OFDM system model suitable for a time-invariant AWGN channel. 
>
> *Transmitter*
>
> ![Transmitter Diagram](https://upload.wikimedia.org/wikipedia/commons/4/4e/OFDM_transmitter_ideal.png)
>
> An OFDM carrier signal is the sum of a number of orthogonal subcarriers, with baseband data on each subcarrier being independently modulated commonly using some type of quadrature amplitude modulation (QAM) or phase-shift keying (PSK). This composite baseband signal is typically used to modulate a main RF carrier.
> s[n] is a serial stream of binary digits. By inverse multiplexing, these are first demultiplexed into N parallel streams, and each one mapped to a (possibly complex) symbol stream using some modulation constellation (QAM, PSK, etc.). Note that the constellations may be different, so some streams may carry a higher bit-rate than others.
>
> An inverse FFT is computed on each set of symbols, giving a set of complex time-domain samples. These samples are then quadrature-mixed to passband in the standard way. The real and imaginary components are first converted to the analogue domain using digital-to-analogue converters (DACs); the analogue signals are then used to modulate cosine and sine waves at the carrier frequency, fc,  respectively. These signals are then summed to give the transmission signal, s(t).
>
> *Receiver*
>
> ![Receiver Diagram](https://upload.wikimedia.org/wikipedia/commons/9/90/OFDM_receiver_ideal.png)
>
> The receiver picks up the signal r(t), which is then quadrature-mixed down to baseband using cosine and sine waves at the carrier frequency. This also creates signals centered on 2fc, so low-pass filters are used to reject these. The baseband signals are then sampled and digitised using analog-to-digital converters (ADCs), and a forward FFT is used to convert back to the frequency domain.
>
> This returns N parallel streams, each of which is converted to a binary stream using an appropriate symbol detector. These streams are then re-combined into a serial stream, s[n], which is an estimate of the original binary stream at the transmitter.
>
> _-- [Wikipedia](https://en.wikipedia.org/wiki/Orthogonal_frequency-division_multiplexing#Idealized_system_model)_
