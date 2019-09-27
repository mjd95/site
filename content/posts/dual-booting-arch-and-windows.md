---
title: "Dual booting Arch and Windows"
date: 2018-06-02T07:13:53+01:00
---

I recently got a [new
laptop](https://www.amazon.co.uk/gp/product/B07CRJ9SHK/ref=oh_aui_detailpage_o03_s00?ie=UTF8&psc=1)
which came with Windows 10 by default. Normally the first thing I do
when I get a new laptop is completely over-write the Windows
installation and install whatever Linux distribution is my favourite at
that point. This time, for gaming reasons, I decided to keep Windows and
dual boot Linux. I've been happily using Arch Linux for a couple of
years now, so I decided to use that as my Linux distribution.

The Arch installation process is mostly straightforward, if a bit time
consuming. The one exception is that using `fdisk` to manage partitions
can be a bit intimidating for inexperienced users like myself. This is
especially true when you're trying to keep a working UEFI Windows
installation in the background, as this involves a number of partitions
in itself.

To save myself any worries, I decided to do the following:

-   Install Ubuntu alongside Windows, using the GUI that provides for
    partitioning to sort out my hard disk.

-   Overwrite that Ubuntu installation with an Arch Linux installation

This worked pretty well. Installing Ubuntu is of course easy. Installing
Arch in an already existing partition is also pretty straightforward.
The only pain point I can remember is that I wasn't sure if I would have
to install a bootloader when I installed Arch. I tried without. When I
booted again I got to the `grub>` screen, so whatever bootloader I was
using didn't know where to look for configuration. I couldn't work out
how to boot in to Arch from there so I just booted from the USB again
and mounted the partition I was trying to install on as usual. I
`chroot`ed in again and set up GRUB correctly, and it worked after that.

Since I had to run through a couple of times, I got very used to
connecting to my WIFI network from the command line. The commands to do
this are

```
iw dev # to get the name of the network interface, e..g `wlp1s0` 
rfkill unblock wifi # may be unnecessary for you
ip link set wlp1s0 up
wpa_passphrase <network_SSID> >> /etc/wpa_supplicant.conf
wpa_supplicant -B -D wext -i wlp1s0 -c /etc/wpa_supplicant.conf
iw wlp1s0 # should look happy
dhclient wlp1s0 # get ip address; hangs if wrong password above
ping www.archlinux.org
```   

I had to do that a few times, which was one of the things that slowed me
down most during the installation. Other than that things went fairly
smoothly!

