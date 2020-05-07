# Network Stack

> The Open Systems Interconnection model (OSI model) is a conceptual model that characterises and standardises the communication functions of a telecommunication or computing system without regard to its underlying internal structure and technology. Its goal is the interoperability of diverse communication systems with standard communication protocols. The model partitions a communication system into abstraction layers.
> A layer serves the layer above it and is served by the layer below it. For example, a layer that provides error-free communications across a network provides the path needed by applications above it, while it calls the next lower layer to send and receive packets that constitute the contents of that path.
> _-- [Wikipedia](https://en.wikipedia.org/wiki/OSI_model)_

This project makes use of the bottom two layers of the model:
* [Layer 0: Physical](/audio_modem/network_stack/physical_layer/)
* [Layer 1: Link](/audio_modem/network_stack/link_layer/)

These layers handle the audio device manipulation and the encoding and modulation respectively.

