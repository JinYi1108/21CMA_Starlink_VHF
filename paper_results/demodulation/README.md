# Packet demodulation results

`event1/` and `event2/` contain the primary LFdemod outputs for the two
Starlink VHF TT&C packets: the decoded payload and frame binaries, per-frame
CSV record, diagnostic image, and JSON summary. Both summaries report a valid
explicit PHY header and payload CRC.

The binary frame contains the decoded payload followed by its two-byte CRC;
the payload binary contains the 81-byte payload only.
