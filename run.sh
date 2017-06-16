#!/usr/bin/env bash
python SpamDetection.py > SpamDetection-log
python positiveSentencesMethod1.py > positiveSentencesMethod1-output
python positiveSentencesMethod2.py > positiveSentencesMethod2-output
python positiveSentencesKNN.py > positiveSentencesKNN-output