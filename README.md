# MRSI Thesis - Robot Action Anticipation for Collaborative Assembly Tasks

MRSI Thesis Repository

## Installation Guide (In Progress)

usb_cam dependency:

```sudo apt-get install libv4l-dev```

other dependencies:

```sudo apt install python3-pip```

## Paper Search Keywords

### Review Paper
combination of keywords was used: (‘‘human–robot
collaborat*’’ OR ‘‘human–robot cooperat*’’ OR ‘‘collaborative robot*’’
OR ‘‘cobot*’’ OR ‘‘hrc’’) AND ‘‘learning’’

This set of keywords
was searched in the title, abstract and keywords records of the journal
articles and conference proceedings written in English, from 2015 to
2020.
Inputting this set of search parameters returned a total of 389 results
from ISI Web of Knowledge (191 articles, 198 proceedings), 178 from
IEEE Xplore (48 articles, 130 proceedings) and 486 from Scopus (206
articles and 280 proceedings).

### Mine

( TITLE-ABS-KEY ( "human–robot collaborat*" ) OR TITLE-ABS-KEY ( human–robot AND cooperat* ) OR TITLE-ABS-KEY ( collaborative AND robot* ) OR TITLE-ABS-KEY ( ‘‘cobot*’’ ) OR TITLE-ABS-KEY ( ‘‘hrc’’ ) AND TITLE-ABS-KEY ( "learning" ) ) AND ( LIMIT-TO ( PUBYEAR , 2023 ) OR LIMIT-TO ( PUBYEAR , 2022 ) OR LIMIT-TO ( PUBYEAR , 2021 ) OR LIMIT-TO ( PUBYEAR , 2020 ) OR LIMIT-TO ( PUBYEAR , 2019 ) OR LIMIT-TO ( PUBYEAR , 2018 ) ) AND ( LIMIT-TO ( LANGUAGE , "english" ) )

## Sensor links

https://www.bosch-sensortec.com/media/boschsensortec/downloads/development_desktop_software/usermanuals/dd2-0_bhyxxx.pdf

## Other thesis in action anticipation

- https://repository.kaust.edu.sa/handle/10754/673882

## Backlog

- antecipate user side with both position and velocity
- learn transformers neural networks
- finish up wrong guess state
- record new demonstration
- fix socket problem
