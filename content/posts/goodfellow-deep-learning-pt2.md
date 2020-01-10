---
title: "Deep Learning Notes Part 2"
date: 2020-01-10T16:20:00+01:00
draft: true
---

Continuing on from the previous one, now focussing on specific problems.

# Chapter 9 - Convolutional Neural Networks

## Introduction

  * Roughly, CNNs are a kind of NN for processing data that has a known grid-like topology.  Most famous for images (a 2D grid of pixels), but also time series (a 1D grid of samples)
  * More precisely, CNNs are just NNs that use convolution in place of matrix multiplication in one of their layers

## The Convolution Operator
