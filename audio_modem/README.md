# Audio Modem

> **Idealised System Model**
>
> This section describes a simple idealized OFDM system model suitable for a time-invariant AWGN channel. 
>
> **Transmitter**
>
> ![Transmitter Diagram](https://upload.wikimedia.org/wikipedia/commons/4/4e/OFDM_transmitter_ideal.png)
>
> An OFDM carrier signal is the sum of a number of orthogonal subcarriers, with baseband data on each subcarrier being independently modulated commonly using some type of quadrature amplitude modulation (QAM) or phase-shift keying (PSK). This composite baseband signal is typically used to modulate a main RF carrier.
> s[n] is a serial stream of binary digits. By inverse multiplexing, these are first demultiplexed into N parallel streams, and each one mapped to a (possibly complex) symbol stream using some modulation constellation (QAM, PSK, etc.). Note that the constellations may be different, so some streams may carry a higher bit-rate than others.
>
> An inverse FFT is computed on each set of symbols, giving a set of complex time-domain samples. These samples are then quadrature-mixed to passband in the standard way. The real and imaginary components are first converted to the analogue domain using digital-to-analogue converters (DACs); the analogue signals are then used to modulate cosine and sine waves at the carrier frequency, fc,  respectively. These signals are then summed to give the transmission signal, s(t).
>
> **Receiver**
>
> ![Receiver Diagram](https://upload.wikimedia.org/wikipedia/commons/9/90/OFDM_receiver_ideal.png)
>
> The receiver picks up the signal r(t), which is then quadrature-mixed down to baseband using cosine and sine waves at the carrier frequency. This also creates signals centered on 2fc, so low-pass filters are used to reject these. The baseband signals are then sampled and digitised using analog-to-digital converters (ADCs), and a forward FFT is used to convert back to the frequency domain.
>
> This returns N parallel streams, each of which is converted to a binary stream using an appropriate symbol detector. These streams are then re-combined into a serial stream, s[n], which is an estimate of the original binary stream at the transmitter.
>
> _-- [Wikipedia](https://en.wikipedia.org/wiki/Orthogonal_frequency-division_multiplexing#Idealized_system_model)_

# Network Stack

> The Open Systems Interconnection model (OSI model) is a conceptual model that characterises and standardises the communication functions of a telecommunication or computing system without regard to its underlying internal structure and technology. Its goal is the interoperability of diverse communication systems with standard communication protocols. The model partitions a communication system into abstraction layers.
>
> A layer serves the layer above it and is served by the layer below it. For example, a layer that provides error-free communications across a network provides the path needed by applications above it, while it calls the next lower layer to send and receive packets that constitute the contents of that path.
>
> _-- [Wikipedia](https://en.wikipedia.org/wiki/OSI_model)_

This project makes use of the bottom two layers of the model:
* Layer 0: Physical
* Layer 1: Link

These layers handle the audio device manipulation and the encoding and modulation respectively.

# Physical Layer

> The physical layer is responsible for the transmission and reception of unstructured raw data between a device and a physical transmission medium. It converts the digital bits into electrical, radio, or optical signals. Layer specifications define characteristics such as voltage levels, the timing of voltage changes, physical data rates, maximum transmission distances, modulation scheme, channel access method and physical connectors. This includes the layout of pins, voltages, line impedance, cable specifications, signal timing and frequency for wireless devices. Bit rate control is done at the physical layer and may define transmission mode as simplex, half duplex, and full duplex. The components of a physical layer can be described in terms of a network topology. Physical layer specifications are included in the specifications for the ubiquitous Bluetooth, Ethernet, and USB standards. An example of a less well-known physical layer specification would be for the CAN standard.
>
> _-- [Wikipedia](https://en.wikipedia.org/wiki/OSI_model)_

# Link Layer
_Sometimes called the "Data Link Layer"_

> The data link layer provides node-to-node data transfer—a link between two directly connected nodes. It detects and possibly corrects errors that may occur in the physical layer. It defines the protocol to establish and terminate a connection between two physically connected devices. It also defines the protocol for flow control between them.
>
> _-- [Wikipedia](https://en.wikipedia.org/wiki/OSI_model)_
