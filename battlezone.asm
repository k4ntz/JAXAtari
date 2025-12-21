; Battlezone for the Atari 2600 VCS
;
; Copyright 1983 Atari
; Written by ?????
;
; Reverse-Engineered by Manuel Polik (cybergoth@nexgo.de)
; Compiles with DASM
;
; History
; 30.10.2.2K      - Started
; 31.10.2.2K      - Finished Crash analysis

    include vcs.h
    processor 6502

; Variables

frameCounter        = $80   ; incremented every frame

tempPointer         = $A2   ; a pointer to various locations
;tempPointer         = $A3

randomVal1          = $BF   ; a more or less random value
randomVal2          = $C0   ; the previous random value
crashCounter        = $C1   ; is 00 or decremented once every frame


; First bank

       ORG $1000
       RORG $D000

       STA $FFF9            ; Start in 2nd bank
      
Bank1Start
       STA WSYNC            ; Finish current line
       STA HMOVE            ;

       LDA #$00             ;
       STA VBLANK           ; Disable VBLANK

       LDA crashCounter     ; In crashing mode?
       BEQ NormalScreen     ; N: Do normal screen
       CMP #$58             ; Y: crash counter < $58?
       BCC ShowCrash        ; Y: show crash screen
       LSR                  ; N: Alternate crash/normal screen
       BCS NormalScreen     ; 

ShowCrash 
       JMP CrashScreen      ;

; Start Normal display

NormalScreen
       LDX    $D4     ;3
       LDA    #$00    ;2
       STA    REFP0   ;3
       LDA    LDC5C,X ;4
       STA    REFP1   ;3
       STA    HMCLR   ;3
       STA    HMP1    ;3
       LDY    LDA00,X ;4
       LDA    #$0E    ;2
       STA    COLUP1  ;3
       STA    COLUPF  ;3
       LDA    #$18    ;2
       STA    GRP0    ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       CPX    #$11    ;2
       BCC    LD044   ;2
       TXA            ;2
       SEC            ;2
       SBC    #$21    ;2
       EOR    #$FF    ;2
       TAX            ;2
LD044: LDA    LDEC2,X ;4
       STA    tempPointer     ;3
       LDA    LDF14,X ;4
       STA    tempPointer+1     ;3
       LDA    LD800,X ;4
       STA    NUSIZ1  ;3
       LDA    frameCounter     ;3
       AND    #$03    ;2
       BNE    LD061   ;2
       LDX    $D4     ;3
       INX            ;2
       TXA            ;2
       AND    #$1F    ;2
       STA    $D4     ;3
LD061: STY    HMP1    ;3
       LDY    #$1E    ;2
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDA    #$24    ;2
       STA    GRP0    ;3
       LDA    (tempPointer),Y ;5
       STA    GRP1    ;3
       LDA    #$02    ;2
       CPY    $D2     ;3
       BNE    LD079   ;2
       STA    ENABL   ;3
LD079: CPY    $D3     ;3
       BNE    LD07F   ;2
       STA    ENAM1   ;3
LD07F: DEY            ;2
       STA    HMCLR   ;3
       LDA    #$30    ;2
       STA    HMP0    ;3
       PHA            ;3
       PLA            ;4
       NOP            ;2
       LDX    #$01    ;2
       STX    NUSIZ0  ;3
       DEX            ;2
       STX    ENABL   ;3
       STX    ENAM1   ;3
LD092: STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDA    LDEDF,Y ;4
       STA    GRP0    ;3
       LDA    (tempPointer),Y ;5
       STA    GRP1    ;3
       LDA    #$02    ;2
       CPY    $D2     ;3
       BEQ    LD0A8   ;2
       NOP            ;2
       BNE    LD0AB   ;2
LD0A8: STA    $011F   ;4
LD0AB: CPY    $D3     ;3
       BEQ    LD0B2   ;2
       NOP            ;2
       BNE    LD0B5   ;2
LD0B2: STA    $011E   ;4
LD0B5: DEY            ;2
       LDA    #$08    ;2
       STA    HMCLR   ;3
       STA    REFP0   ;3
       LDX    #$D0    ;2
       LDA    #$00    ;2
       STA    REFP0   ;3
       STA    ENABL   ;3
       STA    ENAM1   ;3
       CPY    #$01    ;2
       BCS    LD092   ;2
       STX    HMP0    ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDA    #$24    ;2
       STA    GRP0    ;3
       LDA    #$05    ;2
       STA    NUSIZ0  ;3
       LDA    (tempPointer),Y ;5
       STA    GRP1    ;3
       LDA    #$02    ;2
       CPY    $D2     ;3
       BNE    LD0E4   ;2
       STA    ENABL   ;3
LD0E4: CPY    $D3     ;3
       BNE    LD0EA   ;2
       STA    ENAM1   ;3
LD0EA: STA    HMCLR   ;3
       LDA    $BD     ;3
       CMP    #$0A    ;2
       BCS    LD0F6   ;2
       LDA    #$20    ;2
       BNE    LD100   ;2
LD0F6: CMP    #$23    ;2
       BCS    LD0FE   ;2
       LDA    #$10    ;2
       BNE    LD100   ;2
LD0FE: LDA    #$00    ;2
LD100: LDY    $E9     ;3
       STA    $E6     ;3
       LDA    #$18    ;2
       STA    GRP0    ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDX    #$00    ;2
       STX    ENABL   ;3
       STX    ENAM1   ;3
       LDA    $81     ;3
       SEC            ;2
       SBC    $B0     ;3
       CMP    #$20    ;2
       BCS    LD11F   ;2
       CPY    $B2     ;3
       BCC    LD12B   ;2
LD11F: LDA    $81     ;3
       SBC    $B6     ;3
       CMP    #$20    ;2
       BCS    LD131   ;2
       CPY    $B8     ;3
       BCS    LD131   ;2
LD12B: LDA    #$04    ;2
       ORA    $E6     ;3
       STA    $E6     ;3
LD131: STA    WSYNC   ;3
       STA    HMOVE   ;3
       STX    GRP0    ;3
       STX    GRP1    ;3
       STX    REFP0   ;3
       STX    REFP1   ;3
       STX    CTRLPF  ;3
       LDA    #$04    ;2
       STA    COLUPF  ;3
       STA    COLUP1  ;3
       LDA    #$08    ;2
       STA    COLUP0  ;3
       LDA    #$15    ;2
       STA    NUSIZ0  ;3
       LDA    #$35    ;2
       STA    NUSIZ1  ;3
       LDA    $84     ;3
       SEC            ;2
       SBC    #$66    ;2
       BCS    LD15A   ;2
       ADC    #$A0    ;2
LD15A: JSR    PosElement   ;6
       LDA    #$A2    ;2
       STA    COLUBK  ;3
       STY    tempPointer     ;3
       LDA    #$64    ;2
       SEC            ;2
       SBC    $BD     ;3
       STA    $AA     ;3
       LDA    $84     ;3
       SEC            ;2
       SBC    #$56    ;2
       BCS    LD173   ;2
       ADC    #$A0    ;2
LD173: INX            ;2
       JSR    PosElement   ;6
       STY    tempPointer+1     ;3
       LDX    $AF     ;3
       LDA    LD811,X ;4
       STA    $FC     ;3
       LDA    LD81B,X ;4
       STA    $FD     ;3
       LDA    $84     ;3
       LDX    #$03    ;2
       JSR    PosElement   ;6
       STY    $A4     ;3
       LDX    $B5     ;3
       LDA    LD811,X ;4
       STA    $AC     ;3
       LDA    LD81B,X ;4
       STA    $AD     ;3
       LDA    $84     ;3
       LDX    #$02    ;2
       JSR    PosElement   ;6
       PHA            ;3
       PLA            ;4
       PHA            ;3
       PLA            ;4
       LDA    tempPointer     ;3
       STA    HMP0    ;3
       LDA    tempPointer+1     ;3
       STA    HMP1    ;3
       LDA    $A4     ;3
       STA    HMM1    ;3
       STY    HMM0    ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDX    $B4     ;3
       LDA    $BE     ;3
       AND    #$F0    ;2
       CMP    #$30    ;2
       BCS    LD1D2   ;2
       LDA    LDCAF,X ;4
       SEC            ;2
       SBC    $AD     ;3
       STA    $A4     ;3
       LDX    $B5     ;3
       LDA    LD9F6,X ;4
       SBC    #$00    ;2
       BNE    LD204   ;2
LD1D2: BEQ    LD1F8   ;2
       CMP    #$50    ;2
       BEQ    LD1E8   ;2
       LDX    $B5     ;3
       LDA    LD8D4,X ;4
       SEC            ;2
       SBC    $AD     ;3
       STA    $A4     ;3
       LDA    #$DD    ;2
       SBC    #$00    ;2
       BNE    LD204   ;2
LD1E8: LDX    $B5     ;3
       LDA    LD839,X ;4
       SEC            ;2
       SBC    $AD     ;3
       STA    $A4     ;3
       LDA    #$D8    ;2
       SBC    #$00    ;2
       BNE    LD204   ;2
LD1F8: LDA    LD902,X ;4
       SEC            ;2
       SBC    $AD     ;3
       STA    $A4     ;3
       LDA    #$DC    ;2
       SBC    #$00    ;2
LD204: STA    $A5     ;3
       STA    HMCLR   ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDX    $EA     ;3
       LDA    frameCounter     ;3
       AND    #$07    ;2
       BNE    LD22F   ;2
       LDA    $9F     ;3
       BEQ    LD22F   ;2
       LDA    $E8     ;3
       ASL            ;2
       ASL            ;2
       BPL    LD226   ;2
       ASL            ;2
       BMI    LD22F   ;2
       DEX            ;2
       BPL    LD22D   ;2
       LDX    #$01    ;2
LD226: INX            ;2
       CPX    #$03    ;2
       BNE    LD22D   ;2
       LDX    #$00    ;2
LD22D: STX    $EA     ;3
LD22F: STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDA    LDDFC,X ;4
       STA    $FA     ;3
       LDA    LDE02,X ;4
       STA    $EB     ;3
       LDA    #$DE    ;2
       STA    $FB     ;3
       LDY    #$1E    ;2
LD243: DEY            ;2
       LDA    LDA20,Y ;4
       CPY    #$0B    ;2
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       STA    COLUBK  ;3
       BNE    LD243   ;2
       LDA    #$02    ;2
       STA    ENAM0   ;3
LD255: LDA    LDECD,Y ;4
       STA    GRP0    ;3
       STA    GRP1    ;3
       LDA    LDECA,Y ;4
       STA    NUSIZ0  ;3
       STA    HMM0    ;3
       DEY            ;2
       LDA    LDA20,Y ;4
       CPY    #$08    ;2
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       STA    COLUBK  ;3
       BNE    LD255   ;2
       LDA    #$FF    ;2
       STA    GRP0    ;3
       STA    GRP1    ;3
       LDA    #$02    ;2
       STA    ENAM1   ;3
       LDA    #$B0    ;2
       STA    HMP1    ;3
       LDA    #$40    ;2
       STA    HMP0    ;3
       LDA    #$20    ;2
       STA    HMM0    ;3
       LDA    #$E0    ;2
       STA    HMM1    ;3
       LDX    #$84    ;2
       TXS            ;2
       LDA    #$01    ;2
       STA    VDELP0  ;3
       DEY            ;2
LD293: STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDA    LDA20,Y ;4
       STA    COLUBK  ;3
       PLA            ;4
       STA    PF0     ;3
       PLA            ;4
       STA    PF1     ;3
       PLA            ;4
       STA    PF2     ;3
       PLA            ;4
       STA    PF0     ;3
       PLA            ;4
       DEY            ;2
       STA    PF1     ;3
       PLA            ;4
       TXS            ;2
       STA    PF2     ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDA    LDA20,Y ;4
       STA    COLUBK  ;3
       PLA            ;4
       STA    PF0     ;3
       PLA            ;4
       STA    PF1     ;3
       PLA            ;4
       STA    PF2     ;3
       PLA            ;4
       STA    PF0     ;3
       PLA            ;4
       STA    PF1     ;3
       PLA            ;4
       TSX            ;2
       STA    PF2     ;3
       DEY            ;2
       BPL    LD293   ;2
       LDA    #$DE    ;2
       STA    $AB     ;3
       LDA    #$00    ;2
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       STA    GRP0    ;3
       STA    GRP1    ;3
       STA    ENAM0   ;3
       STA    ENAM1   ;3
       LDX    #$04    ;2
       STX    COLUBK  ;3
       STA    PF0     ;3
       STA    PF1     ;3
       STA    PF2     ;3
       LDA    #$20    ;2
       STA    HMBL    ;3
       LDX    #$FF    ;2
       TXS            ;2
       LDA    $EC     ;3
       STA    COLUPF  ;3
       STA    $0114   ;4
       LDA    #$02    ;2
       STA    ENABL   ;3
       LDX    $AE     ;3
       LDA    $BE     ;3
       AND    #$0F    ;2
       CMP    #$03    ;2
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       BCC    LD312   ;2
       LDX    $AF     ;3
       LDA    LDF0A,X ;4
       BPL    LD315   ;2
LD312: LDA    LDE8D,X ;4
LD315: STA    NUSIZ0  ;3
       LDX    $B4     ;3
       LDA    $BE     ;3
       AND    #$F0    ;2
       CMP    #$30    ;2
       BCC    LD328   ;2
       LDX    $B5     ;3
       LDA    LDF0A,X ;4
       BPL    LD32B   ;2
LD328: LDA    LDE8D,X ;4
LD32B: STA    NUSIZ1  ;3
       LDA    $B0     ;3
       CMP    #$11    ;2
       BCS    LD339   ;2
       SBC    #$04    ;2
       BCS    LD339   ;2
       ADC    #$A5    ;2
LD339: STA    HMCLR   ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
LD33F: SBC    #$0F    ;2
       BCS    LD33F   ;2
       EOR    #$07    ;2
       ASL            ;2
       ASL            ;2
       ASL            ;2
       ASL            ;2
       STA    HMP0    ;3
       STA    RESP0   ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDX    $AF     ;3
       LDA    $BE     ;3
       AND    #$0F    ;2
       CMP    #$03    ;2
       BCC    LD370   ;2
       BEQ    LD36B   ;2
       CMP    #$05    ;2
       BEQ    LD366   ;2
       LDA    LDDE8,X ;4
       BNE    LD37C   ;2
LD366: LDA    LD825,X ;4
       BNE    LD37C   ;2
LD36B: LDA    LDDF2,X ;4
       BNE    LD37C   ;2
LD370: CMP    #$02    ;2
       BCC    LD379   ;2
       LDA    LD82F,X ;4
       BNE    LD37C   ;2
LD379: LDA    LDF00,X ;4
LD37C: SEC            ;2
       SBC    $FD     ;3
       STA    $A6     ;3
       LDA    #$DF    ;2
       SBC    #$00    ;2
       STA    $A7     ;3
       STA    HMCLR   ;3
       LDA    $81     ;3
       CMP    #$11    ;2
       BCS    LD395   ;2
       SBC    #$04    ;2
       BCS    LD395   ;2
       ADC    #$A5    ;2
LD395: PHA            ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDX    $B5     ;3
       LDA    $BE     ;3
       AND    #$F0    ;2
       CMP    #$30    ;2
       BCC    LD3B9   ;2
       BNE    LD3AB   ;2
       LDA    LDDF2,X ;4
       BNE    LD3C5   ;2
LD3AB: CMP    #$50    ;2
       BEQ    LD3B4   ;2
       LDA    LDDE8,X ;4
       BNE    LD3C5   ;2
LD3B4: LDA    LD825,X ;4
       BNE    LD3C5   ;2
LD3B9: CMP    #$20    ;2
       BCC    LD3C2   ;2
       LDA    LD82F,X ;4
       BNE    LD3C5   ;2
LD3C2: LDA    LDF00,X ;4
LD3C5: SEC            ;2
       SBC    $AD     ;3
       STA    $A8     ;3
       LDA    #$DF    ;2
       SBC    #$00    ;2
       STA    $A9     ;3
       LDA    $B6     ;3
       CMP    #$11    ;2
       BCS    LD3DC   ;2
       SBC    #$04    ;2
       BCS    LD3DC   ;2
       ADC    #$A5    ;2
LD3DC: LDY    #$34    ;2
       STA    WSYNC   ;3
       STA    HMOVE   ;3
LD3E2: SBC    #$0F    ;2
       BCS    LD3E2   ;2
       EOR    #$07    ;2
       ASL            ;2
       ASL            ;2
       ASL            ;2
       ASL            ;2
       STA    HMP1    ;3
       STA    RESP1   ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDX    $AE     ;3
       LDA    $BE     ;3
       AND    #$0F    ;2
       CMP    #$03    ;2
       BCS    LD40F   ;2
       LDA    LDCAF,X ;4
       SEC            ;2
       SBC    $FD     ;3
       STA    tempPointer     ;3
       LDX    $AF     ;3
       LDA    LD9F6,X ;4
       SBC    #$00    ;2
       BNE    LD441   ;2
LD40F: BNE    LD41F   ;2
       LDA    LD902,X ;4
       SEC            ;2
       SBC    $FD     ;3
       STA    tempPointer     ;3
       LDA    #$DC    ;2
       SBC    #$00    ;2
       BNE    LD441   ;2
LD41F: CMP    #$05    ;2
       BEQ    LD433   ;2
       LDX    $AF     ;3
       LDA    LD8D4,X ;4
       SEC            ;2
       SBC    $FD     ;3
       STA    tempPointer     ;3
       LDA    #$DD    ;2
       SBC    #$00    ;2
       BNE    LD441   ;2
LD433: LDX    $AF     ;3
       LDA    LD839,X ;4
       SEC            ;2
       SBC    $FD     ;3
       STA    tempPointer     ;3
       LDA    #$D8    ;2
       SBC    #$00    ;2
LD441: STA    tempPointer+1     ;3
       STA    HMCLR   ;3
       LDA    #$00    ;2
       STA    ENABL   ;3
       PLA            ;4
       SEC            ;2
       STA    WSYNC   ;3
       STA    HMOVE   ;3
LD44F: SBC    #$0F    ;2
       BCS    LD44F   ;2
       EOR    #$07    ;2
       ASL            ;2
       ASL            ;2
       ASL            ;2
       ASL            ;2
       STA    HMBL    ;3
       STA    RESBL   ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDA    #$0E    ;2
       STA    COLUPF  ;3
       LDA    $E6     ;3
       STA    CTRLPF  ;3
       LDA    $B3     ;3
       STA    REFP0   ;3
       LDA    $B9     ;3
       STA    REFP1   ;3
       STA    HMCLR   ;3
LD473: STA    WSYNC   ;3
       STA    HMOVE   ;3
       CPY    #$2C    ;2
       BEQ    LD4F3   ;2
       DEY            ;2
       CPY    $FC     ;3
       BEQ    LD4A1   ;2
       CPY    $AC     ;3
       BEQ    LD4A7   ;2
       BNE    LD473   ;2
LD486: LDA    (tempPointer),Y ;5
       AND    $B1     ;3
       STA    GRP0    ;3
       LDA    #$00    ;2
       CPY    #$2C    ;2
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       STA    GRP1    ;3
       BEQ    LD514   ;2
       LDA    ($A6),Y ;5
       STA    COLUP0  ;3
       LDA    ($AA),Y ;5
       STA    ENABL   ;3
       DEY            ;2
LD4A1: CPY    $AC     ;3
       BEQ    LD4C5   ;2
       BNE    LD486   ;2
LD4A7: LDA    ($A4),Y ;5
       AND    $B7     ;3
       CPY    CXCLR   ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       STA    GRP1    ;3
       BNE    LD4B8   ;2
       JMP    LD539   ;3
LD4B8: LDA    ($A8),Y ;5
       STA    COLUP1  ;3
       LDA    ($AA),Y ;5
       STA    ENABL   ;3
       DEY            ;2
       CPY    $FC     ;3
       BNE    LD4A7   ;2
LD4C5: LDA    (tempPointer),Y ;5
       AND    $B1     ;3
       STA    GRP0    ;3
       LDA    ($A4),Y ;5
       AND    $B7     ;3
       CPY    #$2C    ;2
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       STA    GRP1    ;3
       BNE    LD4DC   ;2
       JMP    LD55C   ;3
LD4DC: LDA    ($A6),Y ;5
       STA    COLUP0  ;3
       LDA    ($A8),Y ;5
       STA    COLUP1  ;3
       LDA    ($AA),Y ;5
       STA    ENABL   ;3
       DEY            ;2
       BNE    LD4C5   ;2
LD4EB: STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDA    ($FA),Y ;5
       STA    COLUBK  ;3
LD4F3: LDA    ($AA),Y ;5
       STA    ENABL   ;3
       LDA    #$00    ;2
       STA    GRP0    ;3
       STA    GRP1    ;3
       DEY            ;2
       BEQ    LD523   ;2
       BNE    LD4EB   ;2
LD502: LDA    (tempPointer),Y ;5
       AND    $B1     ;3
       STA    GRP0    ;3
       LDA    #$00    ;2
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       STA    GRP1    ;3
       LDA    ($FA),Y ;5
       STA    COLUBK  ;3
LD514: LDA    ($A6),Y ;5
       STA    COLUP0  ;3
       LDA    ($AA),Y ;5
       STA    ENABL   ;3
       DEY            ;2
       CPY    $FD     ;3
       BEQ    LD4EB   ;2
       BNE    LD502   ;2
LD523: LDA    #$DC    ;2
       BNE    LD573   ;2
LD527: LDA    #$00    ;2
       STA    GRP0    ;3
       LDA    ($A4),Y ;5
       AND    $B7     ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       STA    GRP1    ;3
       LDA    ($FA),Y ;5
       STA    COLUBK  ;3
LD539: LDA    ($A8),Y ;5
       STA    COLUP1  ;3
       LDA    ($AA),Y ;5
       STA    ENABL   ;3
       DEY            ;2
LD542: CPY    $AD     ;3
       BEQ    LD4EB   ;2
       BNE    LD527   ;2
LD548: LDA    (tempPointer),Y ;5
       AND    $B1     ;3
       STA    GRP0    ;3
       LDA    ($A4),Y ;5
       AND    $B7     ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       STA    GRP1    ;3
       LDA    ($FA),Y ;5
       STA    COLUBK  ;3
LD55C: LDA    ($A6),Y ;5
       STA    COLUP0  ;3
       LDA    ($A8),Y ;5
       STA    COLUP1  ;3
       LDA    ($AA),Y ;5
       STA    ENABL   ;3
       DEY            ;2
       CPY    $FD     ;3
       BEQ    LD542   ;2
       CPY    $AD     ;3
       BNE    LD548   ;2
       BEQ    LD502   ;2
LD573: STA    tempPointer+1     ;3
       STA    $A5     ;3
       LDA    $BB     ;3
       AND    #$07    ;2
       CLC            ;2
       ADC    #$E4    ;2
       STA    tempPointer     ;3
       LDA    $BC     ;3
       AND    #$07    ;2
       CLC            ;2
       ADC    #$E4    ;2
       STA    $A4     ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDA    #$00    ;2
       STA    GRP0    ;3
       STA    GRP1    ;3
       STA    ENABL   ;3
       LDA    #$17    ;2
       STA    NUSIZ0  ;3
       STA    NUSIZ1  ;3
       LDA    #$04    ;2
       STA    COLUP0  ;3
       STA    RESP0   ;3
       STA    COLUP1  ;3
       LDA    #$10    ;2
       STA    HMP0    ;3
       LDA    #$60    ;2
       STA    HMP1    ;3
       STA    $0112   ;4
       STA    RESM1   ;3
       LDA    #$00    ;2
       STA    HMM0    ;3
       STA    $0111   ;4
       LDA    #$70    ;2
       STA    HMM1    ;3
       LDA    #$C2    ;2
       STA    COLUPF  ;3
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDY    #$0B    ;2
       LDX    #$02    ;2
       STX    ENAM0   ;3
       STX    ENAM1   ;3
       LDA    #$17    ;2
       TAX            ;2
       BNE    LD5DA   ;2
LD5D0: LDA    #$27    ;2
       LDX    #$00    ;2
       CPY    #$04    ;2
       BNE    LD5DA   ;2
       LDX    #$10    ;2
LD5DA: STA    NUSIZ0  ;3
       STA    NUSIZ1  ;3
       STA    HMCLR   ;3
       TXA            ;2
       STX    HMM0    ;3
       BEQ    LD5EB   ;2
       CPY    #$0B    ;2
       BEQ    LD5EB   ;2
       LDA    #$F0    ;2
LD5EB: STA    HMM1    ;3
       DEY            ;2
       BMI    LD5F6   ;2
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       BPL    LD5D0   ;2
LD5F6: LDY    #$05    ;2
       LDA    #$37    ;2
       STA    NUSIZ0  ;3
       LDX    #$01    ;2
       STX    CTRLPF  ;3
       DEX            ;2
       STX    ENAM1   ;3
       STX    REFP0   ;3
       STX    REFP1   ;3
LD607: STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDA    LDED9,Y ;4
       STA    PF2     ;3
       PHA            ;3
       PLA            ;4
       PHA            ;3
       PLA            ;4
       LDX    #$C0    ;2
       PHA            ;3
       PLA            ;4
       PHA            ;3
       PLA            ;4
       STX    COLUBK  ;3
       LDX    $EB     ;3
       NOP            ;2
       NOP            ;2
       NOP            ;2
       DEY            ;2
       STX    COLUBK  ;3
       BMI    LD630   ;2
       CPY    #$02    ;2
       BNE    LD607   ;2
       LDA    #$00    ;2
       STA    ENAM0   ;3
       BEQ    LD607   ;2
LD630: LDA    #$FC    ;2
       STA    GRP0    ;3
       LDY    #$12    ;2
       STA    GRP1    ;3
LD638: STA    WSYNC   ;3
       STA    HMOVE   ;3
       NOP            ;2
       LDA    (tempPointer),Y ;5
       STA    $0106   ;4
       LDA    #$0F    ;2
       NOP            ;2
       STA    PF1     ;3
       LDX    LD8DE,Y ;4
       STX    PF2     ;3
       LDX    #$C0    ;2
       STX    COLUBK  ;3
       LDA    ($A4),Y ;5
       STA    COLUP1  ;3
       STA    HMCLR   ;3
       LDA    LDD7D,Y ;4
       STA    HMP0    ;3
       BEQ    LD65F   ;2
       LDA    #$F0    ;2
LD65F: STA    HMP1    ;3
       LDA    $EB     ;3
       STA    COLUBK  ;3
       DEY            ;2
       BPL    LD638   ;2
       INY            ;2
       STA    WSYNC   ;3
       STA    HMOVE   ;3
       STY    COLUBK  ;3
       STY    GRP0    ;3
       STY    GRP1    ;3
       STY    PF1     ;3
       STY    PF2     ;3

PostCrash
       STY    tempPointer     ;3
       STY    $A4     ;3
       STY    $A6     ;3
       LDA    #$DD    ;2
       STA    tempPointer+1     ;3
       STA    $A5     ;3
       STA    $A7     ;3
       STA    $A9     ;3
       STA    $AB     ;3
       STA    $FD     ;3
       STA    HMCLR   ;3
       LDX    #$10    ;2
       STX    HMP1    ;3
       STA    WSYNC   ;3
       LDA    $9E     ;3
       AND    #$F0    ;2
       LSR            ;2
       STA    $FC     ;3
       LDA    $9E     ;3
       AND    #$0F    ;2
       ASL            ;2
       ASL            ;2
       ASL            ;2
       STA    $AA     ;3
       LDA    $9D     ;3
       AND    #$F0    ;2
       LSR            ;2
       STA    $A8     ;3
       LDY    #$07    ;2
       BIT    $82     ;3
       STA    RESP0   ;3
       STA    RESP1   ;3
       BPL    LD6D3   ;2
       BIT    frameCounter     ;3
       BPL    LD6D3   ;2
       STA    WSYNC   ;3
       LDA    #$58    ;2
       STA    $FC     ;3
       LDA    #$5F    ;2
       STA    $AA     ;3
       LDA    #$65    ;2
       STA    $A8     ;3
       LDA    #$6B    ;2
       STA    $A6     ;3
       LDA    #$71    ;2
       STA    $A4     ;3
       LDA    #$77    ;2
       STA    tempPointer     ;3
       DEY            ;2
LD6D3: STA    WSYNC   ;3
       STA    HMOVE   ;3
       LDA    #$03    ;2
       STA    VDELP1  ;3
       STA    NUSIZ0  ;3
       STA    NUSIZ1  ;3
       LDA    #$00    ;2
       STA    GRP0    ;3
       BIT    $82     ;3
       BPL    LD6EF   ;2
       LDA    frameCounter     ;3
       AND    #$F0    ;2
       ORA    #$04    ;2
       BNE    LD6F1   ;2
LD6EF: LDA    #$B4    ;2
LD6F1: STA    COLUP0  ;3
       STA    COLUP1  ;3
       JSR    LD898   ;6
       LDX    $BA     ;3
       CPX    #$06    ;2
       BCC    LD700   ;2
       LDX    #$06    ;2
LD700: LDA    LD8F6,X ;4
       STA    $FC     ;3
       LDA    LD8F5,X ;4
       STA    $AA     ;3
       LDA    LD8F4,X ;4
       STA    $A8     ;3
       LDA    LD8F3,X ;4
       STA    $A6     ;3
       LDA    LD8F2,X ;4
       STA    $A4     ;3
       LDA    LD8F1,X ;4
       STA    tempPointer     ;3
       STA    WSYNC   ;3
       LDY    #$04    ;2
       JSR    LD898   ;6
       STA    WSYNC   ;3
       JMP    LDFF2   ;3

CrashScreen
       LDA #$01             ;
       STA VDELP0           ; Enable delay for player 0

       LDA crashCounter     ;
       CMP #$20             ; crashCounter >= $20 ?
       BCS DisplayCrash     ; 
       CMP #$18             ; crashCounter < $18 ?
       BCC DisplayBlack     ; 
       LSR                  ; crashCounter = $18 ?
       BCC DisplayCrash     ; 

DisplayBlack
       LDA #$00             ;
       STA COLUBK           ; Blackness
       LDY #$AF             ; Draw $AF black lines 
       JMP FinishScreen     ;

DisplayCrash
       LDA randomVal1       ; Set Crash Pointer:
       STA tempPointer      ; 1. Random Lo-byte
       LDA randomVal2       ;
       AND #$07             ;
       ORA #$D0             ;
       STA tempPointer+1    ; 2. Random Hi-byte between D0-D7

       LDX #$00             ; X -> 0
       LDA (tempPointer),Y  ;
       AND #$7F             ; A -> Random between 00-7F
       STA WSYNC            ; Finish current line
       STA HMOVE            ;
       JSR PosElement       ; Position player 0
       STY HMP0             ; Fine position player 0

       LDA frameCounter     ;
       AND #$F7             ; 
       STA COLUBK           ; Cycle darker colors for the background

       INX                  ; X -> 1
       LDA (tempPointer),Y  ;
       AND #$7F             ; A -> Random between 00-7F
       JSR PosElement       ; Position player 1
       STY HMP1             ; Fine position player 1

       INX                  ; X -> 2
       LDA (tempPointer),Y  ;
       AND #$7F             ; A -> Random between 00-7F
       JSR PosElement       ; Position Missile 0
       STY HMM0             ; Fine position Missile 0

       INX                  ; X -> 2
       LDA (tempPointer),Y  ; 
       AND #$7F             ; A -> Random between 00-7F
       JSR PosElement       ; Position Missile 1
       STY HMM1             ; Fine position Missile 1

       LDA (tempPointer),Y  ;
       TAY                  ; Mu@ge Mix@age Y content
       
       LDA (tempPointer),Y  ;
       STA GRP0             ;
       STA COLUP0           ; Random Shape & Color for Player 0

       DEY                  ; new Pointer to random values

       LDA (tempPointer),Y  ;
       STA GRP1             ;
       STA COLUP1           ; Random Shape & Color for Player 1

       DEY                  ; new Pointer to random values

       LDA (tempPointer),Y  ;
       STA ENAM0            ; Enable Missile 0 or don't :-)

       DEY                  ; new Pointer to random values

       LDA (tempPointer),Y  ;
       STA ENAM1            ; Enable Missile 1 or don't :-)

       STA WSYNC            ;
       STA HMOVE            ; Finish current line

       DEY                  ; new Pointer to random values

       LDA (tempPointer),Y  ;
       STA HMP0             ; Random movement for player 0

       DEY                  ; new Pointer to random values

       LDA (tempPointer),Y  ;
       STA HMP1             ;
       LDA (tempPointer),Y  ;
       STA HMM0             ; Random movement for player 1, missile 0
       LDA (tempPointer),Y  ; & missile 1. 
       STA HMM1             ; (-> Inefficient!!)

       DEY                  ; new Pointer to random values

       LDA (tempPointer),Y  ;
       STA NUSIZ0           ; Random size/# for player/missle 0

       DEY                  ; new Pointer to random values

       LDA (tempPointer),Y  ;
       STA NUSIZ1           ; Random size/# for player/missle 1

       LDY #$A5             ; Yet another $A5 lines to do!

FinishScreen
       DEY                  ;
       STA WSYNC            ;
       STA HMOVE            ; Proceed black/crash display...
       BNE FinishScreen     ; ...until Y expires.
       
       STY COLUBK           ;
       STY GRP0             ;
       STY GRP1             ;
       STY ENAM0            ;
       STY ENAM1            ; prevent post-crash object bleeding :-)
       
       JMP PostCrash        ;

LD7D3: .byte $F0,$05,$BD,$0A,$DF,$10,$03,$BD,$8D,$DE,$4C,$EC,$DF

; Positions an element
; in:   A -> Desired Position
; in:   X -> Element
; out:  Y -> Fine positioning value

PosElement
       CMP #$11                 ; Desired position >= $11
       BCS PositionOk           ; Y:
       SBC #$04                 ; Correct troubles with early RESP
       BCS PositionOk           ;
       ADC #$A5                 ;
PositionOk
       STA WSYNC                ;
       STA HMOVE                ;
.wait
       SBC #$0F                 ;
       BCS .wait                ; RESP loop
       
       EOR #$07                 ;
       ASL                      ;
       ASL                      ;
       ASL                      ;
       ASL                      ;
       TAY                      ; Y-> correct HMXX value
       STA    RESP0,X           ; Position it!
       STA    WSYNC             ;
       STA    HMOVE             ;
       RTS                      ; done, that's all!

LD800: .byte $00,$00,$00,$00,$00,$05,$05,$05,$05,$05,$05,$05,$00,$00,$00,$00
       .byte $00
LD811: .byte $2C,$2C,$2D,$2E,$2E,$2F,$30,$30,$31,$33
LD81B: .byte $2A,$29,$29,$29,$27,$26,$26,$23,$22,$22
LD825: .byte $60,$60,$5F,$5F,$5F,$66,$6F,$79,$86,$86
LD82F: .byte $C8,$C7,$C7,$CE,$CD,$CC,$CB,$D6,$D5,$D5
LD839: .byte $42,$44,$47,$4B,$50,$57,$60,$6A,$77,$86,$18,$18,$3C,$18,$24,$7E
       .byte $00,$3C,$42,$FF,$00,$3C,$C3,$C3,$7E,$00,$18,$18,$7E,$42,$42,$FF
       .byte $00,$18,$18,$18,$7E,$81,$81,$81,$3E,$00,$00,$08,$08,$08,$08,$3E
       .byte $22,$22,$7E,$00,$00,$00,$18,$18,$18,$18,$18,$24,$42,$42,$42,$7F
       .byte $00,$00,$00,$08,$1C,$1C,$1C,$1C,$08,$1C,$22,$41,$41,$41,$FF,$00
       .byte $00,$00,$18,$3C,$3C,$3C,$3C,$18,$3C,$5A,$81,$81,$81,$81,$81
LD898: STY    $AC     ;3
       LDY    $AC     ;3
       LDA    ($FC),Y ;5
       STA    GRP0    ;3
       STA    WSYNC   ;3
LD8A2: LDA    ($AA),Y ;5
       STA    GRP1    ;3
       LDA    ($A8),Y ;5
       STA    GRP0    ;3
       LDA    ($A6),Y ;5
       STA    $AD     ;3
       LDA    ($A4),Y ;5
       TAX            ;2
       LDA    (tempPointer),Y ;5
       TAY            ;2
       LDA    $AD     ;3
       NOP            ;2
       STA    GRP1    ;3
       STX    GRP0    ;3
       STY    GRP1    ;3
       STA    GRP0    ;3
       DEC    $AC     ;5
       LDY    $AC     ;3
       LDA    ($FC),Y ;5
       CPY    #$00    ;2
       STA    GRP0    ;3
       BPL    LD8A2   ;2
       LDX    #$00    ;2
       STX    GRP0    ;3
       STX    GRP1    ;3
       STX    GRP0    ;3
       RTS            ;6

LD8D4: .byte $E5,$CA,$E1,$CD,$DA,$D1,$C1,$B4,$A5,$94
LD8DE: .byte $01,$01,$01,$01,$81,$C1,$F1,$F1,$7D,$3D,$3D,$1F,$1F,$1F,$3D,$3F
       .byte $7F,$F3,$FF
LD8F1: .byte $50
LD8F2: .byte $50
LD8F3: .byte $50
LD8F4: .byte $50
LD8F5: .byte $50
LD8F6: .byte $50,$90,$90,$90,$90,$90,$90,$3A,$AD,$38,$9C,$22
LD902: .byte $7B,$8C,$9D,$0F,$07,$03,$03,$00,$03,$03,$1F,$0F,$1F,$07,$1F,$0F
       .byte $0F,$06,$0F,$0F,$09,$06,$0F,$0F,$0F,$06,$7F,$3E,$1F,$7E,$0C,$33
       .byte $3F,$3F,$12,$0C,$33,$3F,$3F,$1E,$0C,$3F,$3F,$1E,$3F,$04,$3E,$06
       .byte $FF,$7E,$1F,$FE,$0E,$1F,$0E,$1F,$04,$1E,$06,$C3,$FF,$7E,$FF,$18
       .byte $3C,$3C,$1B,$1F,$1F,$0E,$1F,$04,$0A,$0A,$04,$1B,$1F,$1F,$0E,$1F
       .byte $04,$0E,$0E,$04,$7E,$FF,$FF,$7E,$FF,$0C,$0E,$FE,$0E,$1F,$1F,$1F
       .byte $0E,$1F,$04,$06,$1E,$06,$C3,$FF,$7E,$FF,$18,$24,$3C,$FF,$FF,$FF
       .byte $3C,$FF,$18,$1C,$FC,$1C,$3E,$7F,$7F,$7F,$3E,$7F,$0C,$0E,$3E,$0E
       .byte $1F,$3F,$3F,$3F,$1E,$3F,$0C,$1E,$16,$0C,$33,$3F,$3F,$3F,$1E,$3F
       .byte $0C,$1E,$12,$0C,$33,$3F,$3F,$3F,$1E,$3F,$0C,$1E,$1E,$0C,$1E,$3F
       .byte $3F,$3F,$3F,$1E,$3F,$1E,$04,$0E,$3E,$3E,$04,$1E,$3F,$3F,$1E,$3F
       .byte $0C,$0E,$3E,$0E,$1F,$1F,$1F,$1F,$0E,$1F,$0E,$04,$0E,$1E,$1E,$04
       .byte $3E,$7F,$FF,$FF,$FF,$3C,$FF,$3C,$1C,$3E,$7E,$6E,$1C,$C3,$FF,$FF
       .byte $FF,$FF,$3C,$FF,$3C,$18,$3C,$24,$24,$18,$0E,$1F,$1F,$1F,$1F,$0E
       .byte $1F,$0E,$04,$0E,$0E,$0E,$04,$C3,$FF,$FF,$FF,$FF,$3C,$FF,$3C,$18
       .byte $3C,$3C,$3C,$18
LD9F6: .byte $D9,$D9,$D9,$D9,$D9,$D9,$D9,$D9,$DA,$DA
LDA00: .byte $F0,$F0,$F0,$F0,$F0,$40,$40,$40,$40,$40,$40,$40,$F0,$F0,$F0,$F0
       .byte $F0,$10,$10,$10,$10,$60,$60,$60,$60,$60,$60,$60,$10,$10,$10,$10
LDA20: .byte $58,$38,$28,$1A,$0A,$0A,$BA,$AA,$BA,$AA,$A8,$AA,$A8,$A6,$A8,$A6
       .byte $A6,$A6,$A4,$A6,$A4,$A4,$A4,$A4,$A4,$A2,$A4,$A2,$A2,$A2,$3E,$7F
       .byte $7F,$7F,$7F,$7F,$3E,$7F,$3E,$0C,$1E,$7E,$7E,$0E,$04,$00,$1E,$3F
       .byte $3F,$3F,$3F,$1E,$3F,$1E,$0C,$1E,$3E,$3E,$0E,$04,$0E,$1F,$1F,$1F
       .byte $1F,$1F,$0E,$1F,$0E,$04,$0E,$1E,$1E,$0E,$04,$3E,$7F,$FF,$FF,$FF
       .byte $FF,$3C,$FF,$3C,$1C,$3E,$7E,$5E,$3E,$0C,$C3,$FF,$FF,$FF,$FF,$FF
       .byte $3C,$FF,$3C,$18,$3C,$24,$24,$3C,$18,$C3,$FF,$FF,$FF,$FF,$FF,$3C
       .byte $FF,$3C,$18,$3C,$3C,$3C,$3C,$18,$7E,$FF,$FF,$FF,$FF,$FF,$7E,$FF
       .byte $3E,$0C,$1E,$FE,$FE,$FE,$1E,$1E,$0C,$00,$3E,$7F,$7F,$7F,$7F,$3E
       .byte $7F,$1E,$0C,$1E,$7E,$7E,$7E,$1E,$1E,$0C,$00,$1E,$3F,$3F,$3F,$3F
       .byte $1E,$3F,$1E,$04,$0E,$3E,$3E,$3E,$0E,$0E,$04,$00,$1F,$1F,$1F,$1F
       .byte $1F,$0E,$1F,$0E,$04,$0E,$0E,$1E,$1E,$16,$0E,$04,$1B,$1B,$1F,$1F
       .byte $1F,$1F,$0E,$1F,$0E,$04,$0E,$0E,$0A,$0A,$0A,$0E,$04,$1B,$1B,$1F
       .byte $1F,$1F,$1F,$0E,$1F,$0E,$04,$0E,$0E,$0E,$0E,$0E,$0E,$04,$E3,$68
       .byte $02,$02,$02,$02,$02,$02,$02,$02,$02,$02,$02,$02,$02,$02,$02,$03
       .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$03
       .byte $02,$02,$02,$02,$02,$02,$02,$02,$02,$02,$02,$02,$02,$02,$02,$08
       .byte $08,$04,$04,$04,$04,$04,$04,$04,$02,$02,$02,$02,$02,$02,$03,$00
       .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$03,$02
       .byte $02,$02,$02,$02,$02,$04,$04,$04,$04,$04,$04,$04,$08,$00,$00,$10
       .byte $10,$10,$10,$08,$08,$08,$04,$04,$04,$04,$02,$02,$03,$00,$00,$00
       .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$03,$02,$02,$04
       .byte $04,$04,$04,$08,$08,$08,$10,$10,$10,$10,$00,$00,$00,$40,$40,$20
       .byte $20,$20,$10,$10,$08,$08,$04,$04,$02,$03,$00,$00,$00,$00,$00,$00
       .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$03,$02,$04,$04,$08,$08,$10
       .byte $10,$20,$20,$20,$40,$40,$00,$00,$00,$00,$00,$00,$80,$80,$40,$20
       .byte $20,$10,$08,$04,$04,$03,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
       .byte $00,$00,$00,$00,$00,$03,$04,$04,$08,$10,$20,$20,$40,$80,$80,$00
       .byte $00,$00,$00,$00,$00,$00,$00,$00,$40,$00,$20,$00,$10,$08,$04,$00
       .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$04,$00
       .byte $08,$00,$10,$00,$20,$40,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
       .byte $00,$40,$20,$10,$08,$04,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
       .byte $00,$00,$00,$00,$00,$04,$08,$10,$20,$40,$00,$00,$00,$00,$00,$00
       .byte $00,$00,$00,$00,$00,$00,$00,$C0,$30,$0C,$00,$00,$00,$00,$00,$00
       .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$0C,$30,$C0,$00,$00,$00,$00
       .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$FC
LDC5C: .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
       .byte $00,$88,$88,$88,$88,$88,$88,$88,$88,$88,$88,$88,$88,$88,$88,$88
       .byte $1E,$0C,$76,$DA,$A5,$DB,$36,$EF,$5A,$7E,$FB,$5E,$6C,$DA,$34,$6C
       .byte $D4,$14,$08,$30,$16,$45,$8A,$12,$25,$4A,$00,$B1,$44,$40,$92,$24
       .byte $48,$94,$08,$00,$12,$04,$00,$42,$00,$08,$20,$81,$44,$00,$12,$00
       .byte $08,$20,$00
LDCAF: .byte $09,$08,$05,$04,$05,$08,$13,$0F,$0B,$0F,$17,$20,$1B,$31,$1B,$25
       .byte $67,$35,$2A,$35,$3C,$43,$AC,$55,$AC,$4C,$8B,$81,$77,$6D,$5D,$77
       .byte $95,$CE,$C1,$B4,$9F,$B4,$E8,$79,$6A,$5B,$4C,$3D,$5B,$88,$DB,$CA
       .byte $B9,$A8,$97,$B9,$EC,$04,$04,$04,$04,$02,$02,$02,$02,$04,$04,$04
       .byte $04,$02,$02,$02,$02,$04,$04,$04,$04,$02,$02,$02,$02,$04,$04,$04
       .byte $04,$0E,$13,$13,$13,$13,$13,$13,$0E,$1E,$0C,$0C,$0C,$0C,$0C,$0C
       .byte $1C,$0E,$18,$18,$18,$0E,$03,$03,$0E,$1E,$03,$03,$03,$0E,$03,$03
       .byte $1E,$06,$06,$06,$06,$1F,$12,$12,$12,$1E,$03,$03,$03,$1E,$10,$10
       .byte $1E,$0E,$13,$13,$13,$1E,$10,$10,$0E,$02,$02,$02,$07,$07,$02,$02
       .byte $1E,$0E,$13,$13,$13,$0E,$13,$13,$0E,$1E,$03,$03,$03,$1F,$13,$13
       .byte $1E,$00,$00,$00,$00,$00,$00,$00,$00,$06,$09,$16,$14,$16,$09,$06
       .byte $00,$29,$A9,$B9,$A9,$13,$00,$2A,$2A,$3B,$2A,$93,$00,$A3,$A1,$21
       .byte $A3,$A1,$00,$97,$15,$77,$55,$77,$00,$70,$10,$30,$10,$70
LDD7D: .byte $00,$10,$00,$00,$10,$00,$00,$10,$00,$00,$10,$00,$00,$10,$00,$00
       .byte $10,$10,$10,$3E,$3E,$1C,$3F,$38,$7C,$FF,$3E,$00,$00,$00,$08,$18
       .byte $3C,$7E,$FF,$FF,$FF,$7E,$3C,$18,$10,$3C,$7F,$1E,$00,$00,$00,$0C
       .byte $1C,$3E,$7F,$7F,$7F,$3E,$1C,$18,$38,$7E,$1C,$00,$00,$08,$18,$3C
       .byte $7E,$7E,$3C,$18,$10,$18,$3E,$0C,$00,$0C,$1C,$3E,$3E,$1C,$18,$3A
       .byte $10,$FF,$00,$0C,$FF,$30,$FF,$0C,$00,$08,$3C,$FF,$3C,$10,$3C,$00
       .byte $08,$3C,$7E,$3C,$10,$7E,$08,$7E,$10,$08,$08
LDDE8: .byte $24,$24,$28,$27,$2E,$2C,$35,$41,$3F,$4E
LDDF2: .byte $A0,$9F,$9F,$9F,$9D,$9C,$9B,$99,$97,$97
LDDFC: .byte $06,$03,$00,$26,$C4,$C4
LDE02: .byte $D4,$C4,$D4,$C4,$D4,$C4,$D4,$C4,$D4,$C4,$D4,$D4,$D4,$E4,$D4,$E4
       .byte $D4,$E4,$D4,$E4,$D4,$E4,$D4,$E4,$14,$E4,$14,$E4,$14,$E4,$14,$E4
       .byte $F4,$E4,$F4,$E4,$F4,$E4,$F4,$E4,$F4,$24,$F4,$24,$F4,$24,$F4,$24
       .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
       .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
       .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$02
       .byte $02,$02,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
       .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
       .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
LDE8D: .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
       .byte $00,$05,$05,$05,$00,$05,$05,$05,$05,$05,$05,$05,$05,$05,$07,$05
       .byte $05,$05,$05,$07,$07,$07,$05,$05,$05,$07,$07,$07,$07,$05,$07,$07
       .byte $07,$07,$07,$07,$07
LDEC2: .byte $00,$2F,$5D,$8A,$B6,$DF,$06,$2A
LDECA: .byte $4C,$3A,$16
LDECD: .byte $EF,$C6,$9A,$6D,$3F,$10,$35,$25,$15,$7E,$3C,$18
LDED9: .byte $CE,$BE,$7E,$FC,$FC,$FC
LDEDF: .byte $00,$03,$06,$0C,$08,$10,$10,$20,$20,$20,$20,$40,$40,$40,$40,$40
       .byte $40,$40,$40,$40,$20,$20,$21,$21,$12,$12,$0C,$0C,$06,$03,$08,$4E
       .byte $68
LDF00: .byte $A9,$A8,$A8,$AF,$AE,$AD,$AC,$B7,$B6,$B6
LDF0A: .byte $00,$00,$00,$00,$05,$05,$07,$07,$07,$07
LDF14: .byte $DB,$DB,$DB,$DB,$DB,$DB,$DC,$DC,$DC,$DC,$DC,$DB,$DB,$DB,$DB,$DB
       .byte $DB,$02,$46,$48,$02,$02,$42,$46,$48,$02,$02,$02,$02,$40,$42,$46
       .byte $48,$4A,$02,$02,$02,$02,$40,$42,$44,$46,$48,$4A,$02,$02,$02,$02
       .byte $02,$02,$40,$40,$40,$42,$44,$46,$48,$48,$4A,$02,$02,$02,$02,$02
       .byte $02,$40,$40,$40,$40,$42,$44,$46,$46,$48,$48,$4A,$02,$02,$2C,$2C
       .byte $2C,$00,$2C,$02,$02,$2C,$2C,$00,$2C,$2C,$00,$2C,$02,$02,$02,$2C
       .byte $2C,$2C,$00,$2C,$00,$2C,$02,$02,$02,$2C,$2C,$2C,$2C,$00,$2C,$2C
       .byte $2C,$00,$2C,$02,$02,$02,$02,$2C,$2C,$2C,$2C,$2C,$00,$2C,$2C,$2C
       .byte $00,$2C,$00,$2C,$44,$44,$52,$44,$52,$44,$52,$18,$44,$44,$18,$18
       .byte $52,$18,$52,$44,$52,$A2,$A2,$A6,$A6,$A2,$A2,$A2,$A2,$A2,$06,$A6
       .byte $A6,$A6,$A6,$A2,$A2,$A2,$A2,$A2,$A2,$A2,$06,$06,$A6,$A6,$A6,$A6
       .byte $A6,$A6,$A6,$A6,$A2,$A2,$28,$28,$A2,$A2,$A2,$A2,$A2,$08,$28,$28
       .byte $28,$28,$A2,$A2,$A2,$A2,$A2,$A2,$A2,$08,$08,$28,$28,$28,$28,$28
       .byte $28,$28,$28,$F8,$18,$0C,$8C,$C0,$8D,$F9,$FF,$4C,$D3,$D7
LDFF2: STA    $FFF9   ;4
       JMP    Bank1Start   ;3
LDFF8: .byte $88
LDFF9: .byte $00,$12,$A5,$00,$D0,$F6,$7C

; Second bank

       ORG $2000
       RORG $F000

LF000: .byte $00,$28,$50

START2:
       SEI            ;2
       CLD            ;2
       LDX    #$FF    ;2
       TXS            ;2
       LDA    #$00    ;2
LF00A: STA    WSYNC,X ;4
       DEX            ;2
       BNE    LF00A   ;2
       LDX    #$1F    ;2
LF011: LDA    LFDF6,X ;4
       STA    $82,X   ;4
       DEX            ;2
       BPL    LF011   ;2
LF019: LDA    #$24    ;2
       STA    TIM64T  ;4
       INC frameCounter     ;5
       BNE    LF028   ;2
       INC    $E7     ;5
       BNE    LF028   ;2
       DEC    $E7     ;5
LF028: LDA frameCounter     ;3
       ADC randomVal1 ;3
       ADC randomVal2 ;3
       LDY randomVal1 ;3
       STY randomVal2 ;3
       STA randomVal1 ;3
       BIT    $82     ;3
       BPL    LF03C   ;2
       BIT    $3C ;(INPT4)
       BPL    LF041   ;2
LF03C: LSR    SWCHB   ;6
       BCS    LF05B   ;2
LF041: LDA    #$AA    ;2
       STA    $9D     ;3
       STA    $9E     ;3
       LDX    $A1     ;3
       LDA    $EFFF,X ;4
       STA    $A0     ;3
       LDX    #$00    ;2
       JSR    LFB68   ;6
       STX    $82     ;3
       LDA    #$05    ;2
       STA    $BA     ;3
       BNE    LF078   ;2
LF05B: BIT    $82     ;3
       BMI    LF064   ;2
       LDA    SWCHA   ;4
       BNE    LF06F   ;2
LF064: LDA    $A1     ;3
       STA    $BA     ;3
       LDA    frameCounter     ;3
       ASL            ;2
       BNE    LF071   ;2
       LDA randomVal1 ;3
LF06F: STA    $E8     ;3
LF071: LDX    crashCounter     ;3
       BEQ    LF090   ;2
       DEX            ;2
       BNE    LF08E   ;2
LF078: STX    $E1     ;3
       STX    $C9     ;3
       STX    $D1     ;3
       STX    $D0     ;3
       STX    $C8     ;3
       LDA    $BA     ;3
       BNE    LF08E   ;2
       LDA    #$80    ;2
       STA    $82     ;3
       STX    AUDV0   ;3
       STX    AUDV1   ;3
LF08E: STX    crashCounter     ;3
LF090: LDA    #$00    ;2
       STA    $81     ;3
       STA    $BD     ;3
       LDA    $9F     ;3
       BNE    LF09D   ;2
       JMP    LF267   ;3
LF09D: LDA    frameCounter     ;3
       LSR            ;2
       BCC    LF0A5   ;2
       JMP    LF174   ;3
LF0A5: LDA    $9F     ;3
       CMP    #$02    ;2
       BEQ    LF0CA   ;2
       LDA    $E8     ;3
       ASL            ;2
       BIT    $E8     ;3
       BPL    LF0BD   ;2
       BVC    LF0EF   ;2
       ASL            ;2
       BPL    LF121   ;2
       ASL            ;2
       BMI    LF0CA   ;2
       JMP    LF142   ;3
LF0BD: ASL            ;2
       BPL    LF0DE   ;2
       ASL            ;2
       BPL    LF0CD   ;2
       JSR    LF584   ;6
       DEC    $BB     ;5
       INC    $BC     ;5
LF0CA: JMP    LF160   ;3
LF0CD: DEC    $BB     ;5
       LDA    #$02    ;2
       BIT    frameCounter     ;3
       BEQ    LF0D7   ;2
       DEC    $BC     ;5
LF0D7: JSR    LF584   ;6
       LDY    #$1F    ;2
       BNE    LF148   ;2
LF0DE: INC    $BC     ;5
       LDA    #$02    ;2
       BIT    frameCounter     ;3
       BEQ    LF0E8   ;2
       INC    $BB     ;5
LF0E8: JSR    LF584   ;6
       LDY    #$1F    ;2
       BNE    LF127   ;2
LF0EF: ASL            ;2
       BPL    LF110   ;2
       ASL            ;2
       BPL    LF0FF   ;2
       JSR    LF51C   ;6
       INC    $BB     ;5
       DEC    $BC     ;5
       JMP    LF160   ;3
LF0FF: DEC    $BC     ;5
       LDA    #$02    ;2
       BIT    frameCounter     ;3
       BEQ    LF109   ;2
       DEC    $BB     ;5
LF109: JSR    LF51C   ;6
       LDY    #$1F    ;2
       BNE    LF148   ;2
LF110: INC    $BB     ;5
       LDA    #$02    ;2
       BIT    frameCounter     ;3
       BEQ    LF11A   ;2
       INC    $BC     ;5
LF11A: JSR    LF51C   ;6
       LDY    #$1F    ;2
       BNE    LF127   ;2
LF121: INC    $BB     ;5
       INC    $BC     ;5
       LDY    #$7F    ;2
LF127: TYA            ;2
       LDX    #$C5    ;2
       JSR    LF670   ;6
       TYA            ;2
       LDX    #$CD    ;2
       JSR    LF670   ;6
       TYA            ;2
       LDX    #$D8    ;2
       JSR    LF670   ;6
       TYA            ;2
       LDX    #$DE    ;2
       JSR    LF670   ;6
       JMP    LF160   ;3
LF142: DEC    $BB     ;5
       DEC    $BC     ;5
       LDY    #$7F    ;2
LF148: TYA            ;2
       LDX    #$C5    ;2
       JSR    LF67A   ;6
       TYA            ;2
       LDX    #$CD    ;2
       JSR    LF67A   ;6
       TYA            ;2
       LDX    #$D8    ;2
       JSR    LF67A   ;6
       TYA            ;2
       LDX    #$DE    ;2
       JSR    LF67A   ;6
LF160: LDA    $DB     ;3
       LDX    #$DC    ;2
       JSR    LF8F7   ;6
       DEC    $E0     ;5
       BPL    LF171   ;2
       LDA    $E1     ;3
       AND    #$BF    ;2
       STA    $E1     ;3
LF171: JMP    LF267   ;3
LF174: LDA    SWCHB   ;4
       LSR            ;2
       LSR            ;2
       BCC    LF181   ;2
       LDA    #$00    ;2
       STA    $ED     ;3
       BEQ    LF1A2   ;2
LF181: LDA    #$80    ;2
       BIT    $ED     ;3
       BPL    LF18F   ;2
       INC    $ED     ;5
       INC    $ED     ;5
       BIT    $ED     ;3
       BVC    LF1A2   ;2
LF18F: STA    $ED     ;3
       STA    $82     ;3
       STA    AUDV0   ;3
       STA    AUDV1   ;3
       LDX    $A1     ;3
       INX            ;2
       CPX    #$04    ;2
       BNE    LF1A0   ;2
       LDX    #$01    ;2
LF1A0: STX    $A1     ;3
LF1A2: BIT    $E1     ;3
       BVS    LF1D3   ;2
       BIT    $82     ;3
       BPL    LF1B2   ;2
       LDA    $EC     ;3
       CMP    #$2E    ;2
       BEQ    LF1B6   ;2
       BNE    LF1D3   ;2
LF1B2: BIT    $3C ;(INPT4)
       BMI    LF1D3   ;2
LF1B6: LDX    #$04    ;2
       STX    $DE     ;3
       JSR    LFB68   ;6
       LDA    $E1     ;3
       ORA    #$40    ;2
       STA    $E1     ;3
       LDA    #$00    ;2
       STA    $DD     ;3
       STA    $DC     ;3
       STA    $DF     ;3
       LDA    #$80    ;2
       STA    $DB     ;3
       LDA    #$36    ;2
       STA    $E0     ;3
LF1D3: LDX    $C3     ;3
       LDY    $C5     ;3
       JSR    LF788   ;6
       STA    $C7     ;3
       LDA    $D1     ;3
       BEQ    LF1E9   ;2
       LDX    #$C2    ;2
       LDY    #$CA    ;2
       LDA    #$07    ;2
       JSR    LFD86   ;6
LF1E9: LDA    #$FF    ;2
       STA    $D2     ;3
       STA    $D3     ;3
       LDA    $C9     ;3
       BEQ    LF1FE   ;2
       LDX    $C3     ;3
       LDY    $C5     ;3
       JSR    LF967   ;6
       STX    $E2     ;3
       STY    $D2     ;3
LF1FE: LDA    $D1     ;3
       BEQ    LF20D   ;2
       LDX    $CB     ;3
       LDY    $CD     ;3
       JSR    LF967   ;6
       STX    $E3     ;3
       STY    $D3     ;3
LF20D: BIT    $E1     ;3
       BPL    LF267   ;2
       LDA    $D8     ;3
       CMP    #$FF    ;2
       BMI    LF267   ;2
       CMP    #$08    ;2
       BPL    LF240   ;2
       LDA    $D6     ;3
       CMP    #$FC    ;2
       BMI    LF267   ;2
       CMP    #$04    ;2
       BPL    LF267   ;2
       LDA    crashCounter     ;3
       BNE    LF238   ;2
       LDA    #$80    ;2
       STA    crashCounter     ;3
       BIT    $82     ;3
       BMI    LF238   ;2
       LDX    #$02    ;2
       JSR    LFB68   ;6
       DEC    $BA     ;5
LF238: LDA    $E1     ;3
       AND    #$7F    ;2
       STA    $E1     ;3
       BPL    LF267   ;2
LF240: LDX    $D6     ;3
       LDY    $D8     ;3
       JSR    LF6E5   ;6
       BEQ    LF267   ;2
       LDA    $D5     ;3
       BPL    LF24F   ;2
       EOR    #$FF    ;2
LF24F: CMP    #$40    ;2
       LDA    $D8     ;3
       STA    $E9     ;3
       BCS    LF25D   ;2
       JSR    LF852   ;6
       JMP    LF260   ;3
LF25D: JSR    LF85D   ;6
LF260: STA    $BD     ;3
       LDX    #$D6    ;2
       JSR    LFDDA   ;6
LF267: LDY    #$00    ;2
       LDX    #$01    ;2
LF26B: LDA    INTIM   ;4
       BNE    LF26B   ;2
       LDA    #$02    ;2
       STA    VSYNC   ;3

                        
       STA    VBLANK  ; Enable VBLANK
LF276: STA    WSYNC   ;3
       LDA.wy $00E2,Y ;4
       CLC            ;2
       ADC    #$3A    ;2
       SEC            ;2
LF27F: SBC    #$0F    ;2
       BCS    LF27F   ;2
       EOR    #$07    ;2
       ASL            ;2
       ASL            ;2
       ASL            ;2
       ASL            ;2
       STA    HMM1,X  ;4
       STA    RESM1,X ;4
       INY            ;2
       DEX            ;2
       BPL    LF276   ;2
       STY    WSYNC   ;3
       LDA    #$00    ;2
       STA    VSYNC   ;3
       LDX    #$2B    ;2
       STX    TIM64T  ;4
       LDX    #$20    ;2
       STX    HMP0    ;3
       LDX    #$80    ;2
       STX    HMP1    ;3
       STA    VDELP0  ;3
       STA    VDELP1  ;3
       STA    COLUBK  ;3
       LDA    #$05    ;2
       STA    NUSIZ0  ;3
       LDA    #$CA    ;2
       STA    COLUP0  ;3
       STA    $0111   ;4
       STA    RESP0   ;3
       BIT    $82     ;3
       BPL    LF2BE   ;2
       JMP    LF3B4   ;3
LF2BE: LDX    $9F     ;3
       BMI    LF33C   ;2
       LDA    frameCounter     ;3
       CMP    $E5     ;3
       BNE    LF2F1   ;2
       DEC    $E4     ;5
       BPL    LF2EB   ;2
       LDA    #$FF    ;2
       STA    $9F     ;3
       BMI    LF334   ;2
LF2D2: STA    AUDC1   ;3
       TYA            ;2
       LSR            ;2
       LDA    #$0A    ;2
       BCC    LF2DB   ;2
       LSR            ;2
LF2DB: STA    AUDF1   ;3
       LDA    LFEB6,Y ;4
       STA    AUDV1   ;3
       INX            ;2
       LDA    #$08    ;2
       STA    AUDC0   ;3
LF2E7: LDA    #$0F    ;2
       BNE    LF317   ;2
LF2EB: CLC            ;2
       ADC    LFF4A,X ;4
       STA    $E5     ;3
LF2F1: LDY    $E4     ;3
       LDA    #$FE    ;2
       STA    tempPointer+1     ;3
       LDA    LFF4F,X ;4
       CPX    #$02    ;2
       BEQ    LF2D2   ;2
       STA    AUDC0   ;3
       CPX    #$03    ;2
       BEQ    LF2E7   ;2
       CPX    #$01    ;2
       BNE    LF310   ;2
       TYA            ;2
       CMP    #$07    ;2
       BCC    LF30F   ;2
       SBC    #$07    ;2
LF30F: TAY            ;2
LF310: LDA    LFDEB,X ;4
       STA    tempPointer     ;3
       LDA    (tempPointer),Y ;5
LF317: STA    AUDF0   ;3
       BEQ    LF328   ;2
       LDA    #$03    ;2
       CPX    #$00    ;2
       BEQ    LF328   ;2
       LDA    LFDEF,X ;4
       STA    tempPointer     ;3
       LDA    (tempPointer),Y ;5
LF328: STA    AUDV0   ;3
       LDA    #$02    ;2
       CMP    $9F     ;3
       BEQ    LF33A   ;2
       LDA    #$00    ;2
       BEQ    LF338   ;2
LF334: LDA    #$00    ;2
       STA    AUDV0   ;3
LF338: STA    AUDV1   ;3
LF33A: BEQ    LF3B4   ;2
LF33C: LDA    crashCounter     ;3
       BNE    LF39F   ;2
       LDX    $C9     ;3
       CPX    $D1     ;3
       LDA    $C5     ;3
       BCS    LF34C   ;2
       LDX    $D1     ;3
       LDA    $CD     ;3
LF34C: BPL    LF350   ;2
       EOR    #$FF    ;2
LF350: LDY    #$00    ;2
       LSR            ;2
       LSR            ;2
       LSR            ;2
       LSR            ;2
       EOR    #$0F    ;2
       LSR            ;2
       CPX    #$04    ;2
       BCS    LF361   ;2
       STY    AUDV0   ;3
       BCC    LF37D   ;2
LF361: BNE    LF368   ;2
       LSR            ;2
       LDX    #$0D    ;2
       BNE    LF36C   ;2
LF368: LDX    #$01    ;2
       LDA    #$04    ;2
LF36C: STX    AUDC0   ;3
       TAX            ;2
       LDA    frameCounter     ;3
       AND    #$02    ;2
       BEQ    LF376   ;2
       INY            ;2
LF376: STX    AUDV0   ;3
       LDA    LFDF4,Y ;4
       STA    AUDF0   ;3
LF37D: LDY    #$01    ;2
LF37F: LDX    LFDED,Y ;4
       LDA    COLUP1,X;4
       BEQ    LF391   ;2
       LDA    NUSIZ1,X;4
       EOR    #$7F    ;2
       LSR            ;2
       LSR            ;2
       LSR            ;2
       CMP    $D4     ;3
       BEQ    LF3A3   ;2
LF391: DEY            ;2
       BPL    LF37F   ;2
       LDA    $E8     ;3
       CLC            ;2
       ADC    #$10    ;2
       BCS    LF39F   ;2
       LDY    #$01    ;2
       BNE    LF3A5   ;2
LF39F: LDA    #$00    ;2
       BEQ    LF3B2   ;2
LF3A3: LDY    #$00    ;2
LF3A5: LDA    LFFD5,Y ;4
       STA    AUDC1   ;3
       LDA    LFFD6,Y ;4
       STA    AUDF1   ;3
       LDA    LFFFA,Y ;4
LF3B2: STA    AUDV1   ;3
LF3B4: LDX    $9F     ;3
       BNE    LF3C4   ;2
       STX    $B1     ;3
       STX    $B7     ;3
       DEX            ;2
       STX    $D2     ;3
       STX    $D3     ;3
       JMP    LF514   ;3
LF3C4: LDA    frameCounter     ;3
       LSR            ;2
       BCC    LF3CC   ;2
       JMP    LF447   ;3
LF3CC: LDA    $DB     ;3
       LDX    #$DC    ;2
       JSR    LF8F7   ;6
       BIT    $E1     ;3
       BVC    LF3EE   ;2
       LDX    $DC     ;3
       LDY    $DE     ;3
       JSR    LF6E5   ;6
       BEQ    LF3EE   ;2
       LDA    $DE     ;3
       STA    $E9     ;3
       JSR    LF85D   ;6
       STA    $BD     ;3
       LDX    #$DC    ;2
       JSR    LFDDA   ;6
LF3EE: LDY    #$01    ;2
       STY    $A8     ;3
LF3F2: LDX    #$01    ;2
       STX    $A9     ;3
LF3F6: LDA    $E1     ;3
       AND    LFFF0,Y ;4
       BEQ    LF41E   ;2
       BPL    LF405   ;2
       LDA    $DA     ;3
       CMP    #$64    ;2
       BCS    LF41E   ;2
LF405: LDA    LFDED,X ;4
       TAX            ;2
       LDA    LFFFE,Y ;4
       TAY            ;2
       LDA    #$04    ;2
       JSR    LFD86   ;6
       LDY    $A8     ;3
       TXA            ;2
       BMI    LF41E   ;2
       LDA    $E1     ;3
       EOR    LFFF0,Y ;4
       STA    $E1     ;3
LF41E: DEC    $A9     ;5
       LDX    $A9     ;3
       BPL    LF3F6   ;2
       DEC    $A8     ;5
       LDY    $A8     ;3
       BPL    LF3F2   ;2
       LDX    #$C2    ;2
       JSR    LFC85   ;6
       LDX    #$CA    ;2
       JSR    LFC85   ;6
       LDA    INTIM   ;4
       CMP    #$0C    ;2
       BCC    LF444   ;2
       LDX    $CB     ;3
       LDY    $CD     ;3
       JSR    LF788   ;6
       STA    $CF     ;3
LF444: JMP    LF514   ;3
LF447: LDA    $D5     ;3
       LDX    #$D6    ;2
       JSR    LF8F7   ;6
       DEC    $DA     ;5
       BPL    LF458   ;2
       LDA    $E1     ;3
       AND    #$7F    ;2
       STA    $E1     ;3
LF458: LDY    #$00    ;2
       STY    $EC     ;3
       INY            ;2
LF45D: LDX    LFDED,Y ;4
       LDA    COLUP1,X;4
       BEQ    LF476   ;2
       LDA    RSYNC,X ;4
       BMI    LF476   ;2
       LDA    VBLANK,X;4
       CMP    #$02    ;2
       BCC    LF472   ;2
       CMP    #$FE    ;2
       BCC    LF476   ;2
LF472: LDA    #$2E    ;2
       STA    $EC     ;3
LF476: DEY            ;2
       BPL    LF45D   ;2
       LDA    #$00    ;2
       STA    $B1     ;3
       STA    $B7     ;3
       STA    $AE     ;3
       STA    $B4     ;3
       STA    $B2     ;3
       STA    $B8     ;3
       LDA    $C9     ;3
       BEQ    LF4DF   ;2
       LDA    $D1     ;3
       BEQ    LF4F3   ;2
       LDX    $C3     ;3
       LDY    $C5     ;3
       JSR    LF6E5   ;6
       BEQ    LF4DF   ;2
       LDX    $CB     ;3
       LDY    $CD     ;3
       JSR    LF6E5   ;6
       BEQ    LF4FC   ;2
       LDA    $C5     ;3
       CMP    $CD     ;3
       BCS    LF4C3   ;2
       LDX    $C9     ;3
       STX    $BE     ;3
       LDA    $D1     ;3
       ASL            ;2
       ASL            ;2
       ASL            ;2
       ASL            ;2
       ORA    $BE     ;3
       STA    $BE     ;3
       LDX    #$CA    ;2
       LDY    #$B4    ;2
       JSR    LF9AE   ;6
       LDX    #$C2    ;2
       LDY    #$AE    ;2
       JMP    LF504   ;3
LF4C3: LDX    $D1     ;3
       STX    $BE     ;3
       LDA    $C9     ;3
       ASL            ;2
       ASL            ;2
       ASL            ;2
       ASL            ;2
       ORA    $BE     ;3
       STA    $BE     ;3
       LDX    #$C2    ;2
       LDY    #$B4    ;2
       JSR    LF9AE   ;6
       LDX    #$CA    ;2
       LDY    #$AE    ;2
       JMP    LF504   ;3
LF4DF: LDX    $CB     ;3
       LDY    $CD     ;3
       JSR    LF6E5   ;6
       BEQ    LF507   ;2
       LDA    $D1     ;3
       STA    $BE     ;3
       LDX    #$CA    ;2
       LDY    #$AE    ;2
       JMP    LF504   ;3
LF4F3: LDX    $C3     ;3
       LDY    $C5     ;3
       JSR    LF6E5   ;6
       BEQ    LF507   ;2
LF4FC: LDA    $C9     ;3
       STA    $BE     ;3
       LDX    #$C2    ;2
       LDY    #$AE    ;2
LF504: JSR    LF9AE   ;6
LF507: LDA    $C9     ;3
       BEQ    LF514   ;2
       LDX    #$CA    ;2
       LDY    #$C2    ;2
       LDA    #$07    ;2
       JSR    LFD86   ;6
LF514: LDA    INTIM   ;4
       BNE    LF514   ;2
       JMP    LFFF2   ;3
LF51C: LDX    #$C3    ;2
       LDY    #$C5    ;2
       JSR    LF611   ;6
       LDX    #$CB    ;2
       LDY    #$CD    ;2
       JSR    LF611   ;6
       LDX    #$D6    ;2
       LDY    #$D8    ;2
       JSR    LF611   ;6
       LDX    #$DC    ;2
       LDY    #$DE    ;2
       JSR    LF611   ;6
       DEC    $C2     ;5
       DEC    $CA     ;5
       DEC    $D5     ;5
       DEC    $DB     ;5
       LDA    $83     ;3
       LSR            ;2
       LDA    $84     ;3
       ADC    #$02    ;2
       CMP    #$A0    ;2
       BCC    LF54D   ;2
       SBC    #$A0    ;2
LF54D: STA    $84     ;3
       LDA    $83     ;3
       CLC            ;2
       ADC    #$05    ;2
       STA    $83     ;3
       CMP    #$10    ;2
       BCC    LF583   ;2
       SBC    #$08    ;2
       STA    $83     ;3
       LDX    #$12    ;2
LF560: ASL    $85,X   ;6
       ROR    $86,X   ;6
       ROL    $87,X   ;6
       BCC    LF56E   ;2
       LDA    $88,X   ;4
       ORA    #$08    ;2
       STA    $88,X   ;4
LF56E: ASL    $88,X   ;6
       ROR    $89,X   ;6
       ROL    $8A,X   ;6
       BCC    LF57C   ;2
       LDA    $85,X   ;4
       ORA    #$10    ;2
       STA    $85,X   ;4
LF57C: TXA            ;2
       SEC            ;2
       SBC    #$06    ;2
       TAX            ;2
       BPL    LF560   ;2
LF583: RTS            ;6

LF584: LDX    #$C5    ;2
       LDY    #$C3    ;2
       JSR    LF611   ;6
       LDX    #$CD    ;2
       LDY    #$CB    ;2
       JSR    LF611   ;6
       LDY    #$D6    ;2
       LDX    #$D8    ;2
       JSR    LF611   ;6
       LDY    #$DC    ;2
       LDX    #$DE    ;2
       JSR    LF611   ;6
       INC    $C2     ;5
       INC    $CA     ;5
       INC    $D5     ;5
       INC    $DB     ;5
       LDA    $83     ;3
       LSR            ;2
       LDA    $84     ;3
       SBC    #$02    ;2
       CMP    #$A0    ;2
       BCC    LF5B6   ;2
       CLC            ;2
       ADC    #$A0    ;2
LF5B6: STA    $84     ;3
       LDA    $83     ;3
       SEC            ;2
       SBC    #$05    ;2
       STA    $83     ;3
       CMP    #$01    ;2
       BEQ    LF5C5   ;2
       BPL    LF601   ;2
LF5C5: CLC            ;2
       ADC    #$08    ;2
       STA    $83     ;3
       LDX    #$12    ;2
LF5CC: LSR    $8A,X   ;6
       ROL    $89,X   ;6
       ROR    $88,X   ;6
       LDA    $88,X   ;4
       AND    #$08    ;2
       BEQ    LF5E1   ;2
       SEC            ;2
       LDA    $88,X   ;4
       AND    #$F0    ;2
       STA    $88,X   ;4
       BCS    LF5E2   ;2
LF5E1: CLC            ;2
LF5E2: ROR    $87,X   ;6
       ROL    $86,X   ;6
       ROR    $85,X   ;6
       LDA    $85,X   ;4
       AND    #$08    ;2
       BEQ    LF5FA   ;2
       LDA    $8A,X   ;4
       ORA    #$80    ;2
       STA    $8A,X   ;4
       LDA    $85,X   ;4
       AND    #$F0    ;2
       STA    $85,X   ;4
LF5FA: TXA            ;2
       SEC            ;2
       SBC    #$06    ;2
       TAX            ;2
       BPL    LF5CC   ;2
LF601: RTS            ;6

LF602: INY            ;2
       INY            ;2
       TYA            ;2
       LSR            ;2
       CMP    #$03    ;2
       BNE    LF60C   ;2
       LDA    #$02    ;2
LF60C: TAY            ;2
       ASL            ;2
       ASL            ;2
       ASL            ;2
       RTS            ;6

LF611: LDA    VSYNC,X ;4
       STA    $A5     ;3
       LDA.wy $0000,Y ;4
       STA    $A7     ;3
       LDA    #$00    ;2
       ASL    $A7     ;5
       BCC    LF622   ;2
       LDA    #$FF    ;2
LF622: STA    $A6     ;3
       STY    $A9     ;3
       LDY    #$A6    ;2
       JSR    LF6FD   ;6
       ASL    $A7     ;5
       ROL    $A6     ;5
       JSR    LF6FD   ;6
       TXA            ;2
       LDX    $A9     ;3
       STA    $A9     ;3
       LDA    #$00    ;2
       ASL    $A5     ;5
       BCC    LF63F   ;2
       LDA    #$FF    ;2
LF63F: STA    $A4     ;3
       LDY    #$A4    ;2
       JSR    LF70D   ;6
       ASL    $A5     ;5
       ROL    $A4     ;5
       JSR    LF70D   ;6
       LDA    $A6     ;3
       ASL    $A7     ;5
       ROL            ;2
       ASL    $A7     ;5
       ROL            ;2
       BMI    LF65D   ;2
       JSR    LF67A   ;6
       JMP    LF662   ;3
LF65D: EOR    #$FF    ;2
       JSR    LF670   ;6
LF662: LDX    $A9     ;3
       LDA    $A4     ;3
       ASL    $A5     ;5
       ROL            ;2
       ASL    $A5     ;5
       ROL            ;2
       BPL    LF67A   ;2
       EOR    #$FF    ;2
LF670: CLC            ;2
       ADC    VBLANK,X;4
       STA    VBLANK,X;4
       BCC    LF679   ;2
       INC    VSYNC,X ;6
LF679: RTS            ;6

LF67A: SEC            ;2
       EOR    #$FF    ;2
       ADC    VBLANK,X;4
       STA    VBLANK,X;4
       BCS    LF685   ;2
       DEC    VSYNC,X ;6
LF685: RTS            ;6

LF686: BIT    $82     ;3
       BMI    LF6E4   ;2
       STA    $AC     ;3
       LDX    #$01    ;2
LF68E: LDA    $9D,X   ;4
       AND    #$F0    ;2
       CMP    #$A0    ;2
       BNE    LF6A7   ;2
       LDA    $9D,X   ;4
       AND    #$0F    ;2
       STA    $9D,X   ;4
       CMP    #$0A    ;2
       BNE    LF6A7   ;2
       LDA    #$00    ;2
       STA    $9D,X   ;4
       DEX            ;2
       BPL    LF68E   ;2
LF6A7: SED            ;2
       CLC            ;2
       LDA    $AC     ;3
       ADC    $9D     ;3
       STA    $9D     ;3
       BCC    LF6C4   ;2
       LDA    $9E     ;3
       CMP    #$11    ;2
       BCS    LF6C4   ;2
       AND    #$0F    ;2
       CMP    #$04    ;2
       BEQ    LF6C1   ;2
       CMP    #$09    ;2
       BNE    LF6C3   ;2
LF6C1: INC    $BA     ;5
LF6C3: SEC            ;2
LF6C4: LDA    #$00    ;2
       ADC    $9E     ;3
       STA    $9E     ;3
       CLD            ;2
       LDX    #$01    ;2
LF6CD: LDA    $9D,X   ;4
       AND    #$F0    ;2
       BNE    LF6E4   ;2
       LDA    $9D,X   ;4
       ORA    #$A0    ;2
       STA    $9D,X   ;4
       AND    #$0F    ;2
       BNE    LF6E4   ;2
       LDA    #$AA    ;2
       STA    $9D,X   ;4
       DEX            ;2
       BPL    LF6CD   ;2
LF6E4: RTS            ;6

LF6E5: TYA            ;2
       BMI    LF6FA   ;2
       TXA            ;2
       BPL    LF6EF   ;2
       EOR    #$FF    ;2
       TAX            ;2
       INX            ;2
LF6EF: STX    $A4     ;3
       INY            ;2
       INY            ;2
       CPY    $A4     ;3
       BCC    LF6FA   ;2
       LDA    #$FF    ;2
       RTS            ;6

LF6FA: LDA    #$00    ;2
       RTS            ;6

LF6FD: LDA    VBLANK,X;4
       CLC            ;2
       ADC.wy $0001,Y ;4
       STA    VBLANK,X;4
       LDA    VSYNC,X ;4
       ADC.wy $0000,Y ;4
       STA    VSYNC,X ;4
       RTS            ;6

LF70D: LDA    VBLANK,X;4
       SEC            ;2
       SBC.wy $0001,Y ;4
       STA    VBLANK,X;4
       LDA    VSYNC,X ;4
       SBC.wy $0000,Y ;4
       STA    VSYNC,X ;4
       RTS            ;6

LF71D: LDA    VBLANK,X;4
       STA    $A5     ;3
       LDA    RSYNC,X ;4
       STA    $A6     ;3
       LDA    VSYNC,X ;4
       STA    $A4     ;3
       BPL    LF735   ;2
       EOR    #$FF    ;2
       STA    $A4     ;3
       LDA    $A5     ;3
       EOR    #$FF    ;2
       STA    $A5     ;3
LF735: LDA    WSYNC,X ;4
       BEQ    LF749   ;2
LF739: ASL    $A5     ;5
       ROL    $A4     ;5
       BMI    LF746   ;2
       ASL    $A6     ;5
       ROL            ;2
       BPL    LF739   ;2
       BMI    LF749   ;2
LF746: ASL    $A6     ;5
       ROL            ;2
LF749: STA    $A6     ;3
       LDA    VSYNC,X ;4
       LDX    $A4     ;3
       STA    $A4     ;3
       LDA    #$80    ;2
       STA    $A5     ;3
       LDY    #$00    ;2
LF757: TXA            ;2
       SEC            ;2
       SBC    $A6     ;3
       BCC    LF762   ;2
       TAX            ;2
       TYA            ;2
       ORA    $A5     ;3
       TAY            ;2
LF762: TXA            ;2
       ASL            ;2
       BCS    LF769   ;2
       TAX            ;2
       BCC    LF76B   ;2
LF769: LSR    $A6     ;5
LF76B: LSR    $A5     ;5
       BNE    LF757   ;2
       STY    $A6     ;3
       TYA            ;2
       LSR            ;2
       LSR            ;2
       SEC            ;2
       ADC    $A6     ;3
       LSR            ;2
       LDY    $A4     ;3
       BPL    LF784   ;2
       STA    $A6     ;3
       LDA    #$4D    ;2
       SEC            ;2
       SBC    $A6     ;3
       RTS            ;6

LF784: CLC            ;2
       ADC    #$4D    ;2
       RTS            ;6

LF788: TXA            ;2
       BNE    LF794   ;2
       TYA            ;2
       BMI    LF791   ;2
       LDA    #$00    ;2
       RTS            ;6

LF791: LDA    #$80    ;2
       RTS            ;6

LF794: LDA    #$C0    ;2
       STA    $A8     ;3
       TXA            ;2
       EOR    #$80    ;2
       STA    $A5     ;3
       TYA            ;2
       EOR    #$FF    ;2
       CLC            ;2
       ADC    #$01    ;2
       STA    $A6     ;3
       EOR    #$80    ;2
       CMP    $A5     ;3
       BCC    LF7B8   ;2
       LDY    $A6     ;3
       TXA            ;2
       EOR    #$FF    ;2
       CLC            ;2
       ADC    #$01    ;2
       TAX            ;2
       LDA    #$40    ;2
       STA    $A8     ;3
LF7B8: TXA            ;2
       EOR    #$80    ;2
       STA    $A5     ;3
       TYA            ;2
       EOR    #$80    ;2
       CMP    $A5     ;3
       BCC    LF7D7   ;2
       LDA    $A8     ;3
       CLC            ;2
       ADC    #$40    ;2
       STA    $A8     ;3
       TYA            ;2
       STX    $A5     ;3
       TAX            ;2
       LDA    $A5     ;3
       EOR    #$FF    ;2
       CLC            ;2
       ADC    #$01    ;2
       TAY            ;2
LF7D7: TYA            ;2
       BPL    LF7E7   ;2
       EOR    #$FF    ;2
       TAY            ;2
       INY            ;2
       JSR    LF7EE   ;6
       EOR    #$FF    ;2
       SEC            ;2
       ADC    $A8     ;3
       RTS            ;6

LF7E7: JSR    LF7EE   ;6
       CLC            ;2
       ADC    $A8     ;3
       RTS            ;6

LF7EE: TXA            ;2
       LSR            ;2
       BCC    LF7F8   ;2
       ROL            ;2
       TAX            ;2
       TYA            ;2
       ASL            ;2
       TAY            ;2
       TXA            ;2
LF7F8: STA    $A5     ;3
       LDX    #$00    ;2
       LDA    #$80    ;2
       STA    $A6     ;3
LF800: TYA            ;2
       SEC            ;2
       SBC    $A5     ;3
       BCC    LF80B   ;2
       TAY            ;2
       TXA            ;2
       ORA    $A6     ;3
       TAX            ;2
LF80B: LSR    $A5     ;5
       BCC    LF814   ;2
       ROL    $A5     ;5
       TYA            ;2
       ASL            ;2
       TAY            ;2
LF814: LSR    $A6     ;5
       BNE    LF800   ;2
       LDY    #$10    ;2
       LDA    #$B4    ;2
       STA    $A6     ;3
       LDA    #$08    ;2
LF820: STA    $A4     ;3
       LDA    #$FF    ;2
       STA    $A7     ;3
       TXA            ;2
LF827: CMP    ($A6),Y ;5
       BCC    LF834   ;2
       BEQ    LF850   ;2
       TYA            ;2
       CLC            ;2
       ADC    $A4     ;3
       TAY            ;2
       BPL    LF839   ;2
LF834: TYA            ;2
       SEC            ;2
       SBC    $A4     ;3
       TAY            ;2
LF839: TXA            ;2
       LSR    $A4     ;5
       BNE    LF827   ;2
LF83E: CMP    ($A6),Y ;5
       BCS    LF847   ;2
       DEY            ;2
       BPL    LF83E   ;2
       LDY    #$00    ;2
LF847: CMP    ($A6),Y ;5
       BCC    LF850   ;2
       BEQ    LF850   ;2
       INY            ;2
       BPL    LF847   ;2
LF850: TYA            ;2
       RTS            ;6

LF852: TAX            ;2
       LDA    #$52    ;2
       STA    $A6     ;3
       LDY    #$19    ;2
       LDA    #$0D    ;2
       BNE    LF820   ;2
LF85D: TAX            ;2
       LDA    #$85    ;2
       STA    $A6     ;3
       LDY    #$17    ;2
       LDA    #$0C    ;2
       BNE    LF820   ;2
LF868: LDX    $A8     ;3
       LDA    NUSIZ1,X;4
       CMP    #$08    ;2
       LDA    WSYNC,X ;4
       LDX    $A4     ;3
       LDY    $A5     ;3
       BCS    LF87D   ;2
       STX    $A4     ;3
       ADC    $A4     ;3
       JMP    LF88C   ;3
LF87D: STA    $A4     ;3
       JSR    LF602   ;6
       EOR    #$FF    ;2
       SEC            ;2
       ADC    $A4     ;3
       STX    $A4     ;3
       SEC            ;2
       SBC    $A4     ;3
LF88C: LDY    $A5     ;3
       TAX            ;2
       JSR    LF602   ;6
       STA    $A4     ;3
       TXA            ;2
       CMP    #$D0    ;2
       BCS    LF8BC   ;2
       CMP    #$A0    ;2
       BCC    LF8A3   ;2
       LDA    #$9F    ;2
       LDX    #$00    ;2
       BEQ    LF8D3   ;2
LF8A3: ADC    $A4     ;3
       STY    $A4     ;3
       LDY    #$FF    ;2
       STY    $AD     ;3
LF8AB: SEC            ;2
       SBC    $A4     ;3
       CMP    #$A0    ;2
       BCC    LF8B6   ;2
       ASL    $AD     ;5
       BMI    LF8AB   ;2
LF8B6: TXA            ;2
       LDX    $AD     ;3
       JMP    LF8D3   ;3
LF8BC: SBC    #$60    ;2
       TAX            ;2
       STY    $A4     ;3
       LDY    #$7F    ;2
       STY    $AD     ;3
LF8C5: CLC            ;2
       ADC    $A4     ;3
       CMP    #$A0    ;2
       BCS    LF8D0   ;2
       LSR    $AD     ;5
       BNE    LF8C5   ;2
LF8D0: TXA            ;2
       LDX    $AD     ;3
LF8D3: LDY    $A8     ;3
       STA.wy $0002,Y ;5
       LDA.wy $0005,Y ;4
       BNE    LF8DE   ;2
       RTS            ;6

LF8DE: TXA            ;2
       BMI    LF8EC   ;2
       LSR            ;2
       BCS    LF8E5   ;2
       RTS            ;6

LF8E5: TXA            ;2
       CLC            ;2
LF8E7: ROR            ;2
       BCS    LF8E7   ;2
       TAX            ;2
       RTS            ;6

LF8EC: TXA            ;2
       LSR            ;2
       BCC    LF8F1   ;2
       RTS            ;6

LF8F1: TXA            ;2
LF8F2: ROL            ;2
       BCS    LF8F2   ;2
       TAX            ;2
       RTS            ;6

LF8F7: STX    $A5     ;3
       CMP    #$80    ;2
       BCS    LF925   ;2
       CMP    #$40    ;2
       BCS    LF912   ;2
       TAY            ;2
       LDA    LFEFA,Y ;4
       LDX    $A5     ;3
       JSR    LF670   ;6
       JSR    LF94F   ;6
       INX            ;2
       INX            ;2
       JMP    LF67A   ;3
LF912: SBC    #$3F    ;2
       TAY            ;2
       LDA    LFEFA,Y ;4
       LDX    $A5     ;3
       INX            ;2
       INX            ;2
       JSR    LF670   ;6
       JSR    LF94F   ;6
       JMP    LF670   ;3
LF925: CMP    #$C0    ;2
       BCS    LF93C   ;2
       SBC    #$7F    ;2
       TAY            ;2
       LDA    LFEFA,Y ;4
       LDX    $A5     ;3
       JSR    LF67A   ;6
       JSR    LF94F   ;6
       INX            ;2
       INX            ;2
       JMP    LF670   ;3
LF93C: SBC    #$C0    ;2
       TAY            ;2
       LDA    LFEFA,Y ;4
       LDX    $A5     ;3
       INX            ;2
       INX            ;2
       JSR    LF67A   ;6
       JSR    LF94F   ;6
       JMP    LF67A   ;3
LF94F: STY    $A6     ;3
       LDA    #$40    ;2
       SEC            ;2
       SBC    $A6     ;3
       TAY            ;2
       LDA    LFEFA,Y ;4
       LDX    $A5     ;3
       RTS            ;6

LF95D: LSR            ;2
       LSR            ;2
       LSR            ;2
       CMP    #$10    ;2
       BCC    LF966   ;2
       ORA    #$E0    ;2
LF966: RTS            ;6

LF967: TYA            ;2
       JSR    LF95D   ;6
       CLC            ;2
       ADC    #$0F    ;2
       TAY            ;2
       TXA            ;2
       JSR    LF95D   ;6
       STA    $A5     ;3
       LSR            ;2
       LSR            ;2
       CMP    #$20    ;2
       BCC    LF97D   ;2
       ORA    #$C0    ;2
LF97D: EOR    #$FF    ;2
       SEC            ;2
       ADC    $A5     ;3
       TAX            ;2
       BPL    LF98A   ;2
       EOR    #$FF    ;2
       SEC            ;2
       ADC    #$00    ;2
LF98A: CLC            ;2
       SBC    LFE4B,Y ;4
       BPL    LF998   ;2
       TYA            ;2
       BMI    LF998   ;2
       CMP    #$1F    ;2
       BPL    LF998   ;2
       RTS            ;6

LF998: LDY    #$FF    ;2
       LDX    #$00    ;2
       RTS            ;6

LF99D: .byte $B5,$01,$10,$02,$49,$FF,$85,$A4,$B5,$03,$10,$02,$49,$FF,$65,$A4
       .byte $60
LF9AE: STY    $A8     ;3
       STX    $A7     ;3
       LDA    RSYNC,X ;4
       STA.wy $0004,Y ;5
       INX            ;2
       JSR    LF71D   ;6
       LDX    $A8     ;3
       STA    WSYNC,X ;4
       LDX    $A7     ;3
       LDA    VBLANK,X;4
       BPL    LF9C7   ;2
       EOR    #$FF    ;2
LF9C7: LSR            ;2
       LSR            ;2
       STA    $A4     ;3
       LSR            ;2
       LSR            ;2
       CLC            ;2
       ADC    $A4     ;3
       STA    $A4     ;3
       LSR            ;2
       LSR            ;2
       CLC            ;2
       ADC    $A4     ;3
       CLC            ;2
       ADC    RSYNC,X ;4
       CMP    #$7F    ;2
       BCS    LF9F8   ;2
       LDX    #$09    ;2
LF9E0: CMP    LFED5,X ;4
       BCC    LF9E8   ;2
       DEX            ;2
       BNE    LF9E0   ;2
LF9E8: LDY    $A8     ;3
       STX    VBLANK,Y;4
       LDX    $A7     ;3
       LDA    COLUP1,X;4
       BEQ    LF9F8   ;2
       CMP    #$03    ;2
       BCC    LFA06   ;2
       BCS    LFA55   ;2
LF9F8: LDA    #$00    ;2
       LDY    $A8     ;3
       STA.wy $0000,Y ;5
       STA.wy $0001,Y ;5
       STA.wy $0002,Y ;5
       RTS            ;6

LFA06: LDY    $A8     ;3
       LDX    VBLANK,Y;4
       LDY    $A7     ;3
       LDA.wy $0000,Y ;4
       SEC            ;2
       SBC.wy $0005,Y ;4
       STA    $A5     ;3
       BPL    LFA19   ;2
       EOR    #$FF    ;2
LFA19: LDY    LFE95,X ;4
       STY    $A4     ;3
       CPX    #$00    ;2
       BEQ    LFA2F   ;2
       LDY    LFEDE,X ;4
LFA25: CMP    LFEE8,Y ;4
       BCC    LFA2F   ;2
       INC    $A4     ;5
       INY            ;2
       BCS    LFA25   ;2
LFA2F: LDA    $A4     ;3
       LDY    $A8     ;3
       STA.wy $0000,Y ;5
       LDA    #$00    ;2
       BIT    $A5     ;3
       BMI    LFA3E   ;2
       LDA    #$08    ;2
LFA3E: STA.wy $0005,Y ;5
       LDX    VSYNC,Y ;4
       LDA    LFE16,X ;4
       STA    $A4     ;3
       LDA    #$00    ;2
       JSR    LFFEC   ;6
       STA    $A5     ;3
       JSR    LF868   ;6
       STX    RSYNC,Y ;4
       RTS            ;6

LFA55: LDX    $A8     ;3
       LDY    $A7     ;3
       LDA.wy $0006,Y ;4
       STA    VSYNC,X ;4
       LDX    #$00    ;2
       LDA.wy $0000,Y ;4
       BMI    LFA67   ;2
       LDX    #$08    ;2
LFA67: LDY    $A8     ;3
       STX    NUSIZ1,Y;4
       LDX    VBLANK,Y;4
       LDA    LFF3B,X ;4
       STA    $A4     ;3
       LDA    #$01    ;2
       JSR    LFFEC   ;6
       STA    $A5     ;3
       JSR    LF868   ;6
       STX    RSYNC,Y ;4
       LDX    $A7     ;3
       LDA    COLUP1,X;4
       CMP    #$03    ;2
       BNE    LFA91   ;2
       LDX    VBLANK,Y;4
       LDA    LFE70,X ;4
       AND.wy $0003,Y ;4
       STA.wy $0003,Y ;5
LFA91: RTS            ;6

LFA92: LDA randomVal1 ;3
       AND    #$03    ;2
       ADC    #$01    ;2
       ADC    $A0     ;3
       BCS    LFA9E   ;2
       STA    $A0     ;3
LFA9E: RTS            ;6

LFA9F: .byte $86,$AD,$F6,$06,$A5,$80,$D0,$09,$A5,$E7,$29,$0F,$D0,$03,$20,$92
       .byte $FA,$20,$64,$FD,$F0,$07,$20,$81,$FD,$F0,$E4,$D0,$0A,$A5,$C9,$05
       .byte $D1,$F0,$04,$B5,$06,$D0,$D8,$A4,$C1,$D0,$D4,$A9,$40,$C5,$C8,$F0
       .byte $CE,$C5,$D0,$F0,$CA,$A5,$BF,$2A,$95,$00,$2A,$45,$C0,$85,$A4,$C9
       .byte $80,$6A,$95,$01,$A5,$A4,$2A,$45,$C0,$C9,$80,$6A,$95,$03,$20,$9D
       .byte $F9,$A6,$AD,$C9,$32,$B0,$04,$A9,$50,$95,$03,$A5,$C3,$E5,$CB,$10
       .byte $02,$49,$FF,$85,$A4,$A5,$C5,$E5,$CD,$10,$02,$49,$FF,$65,$A4,$C9
       .byte $10,$90,$8C,$A9,$40,$95,$06,$A5,$C0,$29,$03,$D0,$30,$A9,$04,$C5
       .byte $C9,$F0,$2A,$C5,$D1,$F0,$26,$A4,$A0,$C0,$0F,$90,$38,$A5,$80,$29
       .byte $04,$F0,$16,$A9,$05,$C5,$C9,$F0,$10,$C5,$D1,$F0,$0C,$A9,$7F,$95
       .byte $03,$A9,$00,$95,$01,$A9,$05,$D0,$1C,$A9,$04,$D0,$18,$A9,$01,$A4
       .byte $A0,$C0,$7F,$90,$10,$C0,$F0,$B0,$0A,$A0,$02,$C4,$C9,$F0,$06,$C4
       .byte $D1,$F0,$02,$A9,$02,$95,$07,$A2,$01
LFB68: LDA    $9F     ;3
       CMP    #$02    ;2
       BEQ    LFB7D   ;2
       STX    $9F     ;3
       LDA    LFF45,X ;4
       STA    $E4     ;3
       LDA    frameCounter     ;3
       CLC            ;2
       ADC    LFF4A,X ;4
       STA    $E5     ;3
LFB7D: RTS            ;6

LFB7E: .byte $86,$AD,$B4,$06,$88,$10,$02,$A0,$00,$94,$06,$20,$9D,$F9,$C9,$1E
       .byte $90,$4A,$B4,$07,$C0,$02,$D0,$03,$4C,$28,$FC,$C9,$46,$B0,$5A,$20
       .byte $F7,$FB,$A6,$AD,$B5,$05,$38,$F5,$00,$C9,$01,$90,$04,$C9,$FF,$90
       .byte $2A,$24,$E1,$30,$26,$B5,$06,$D0,$22,$A5,$C1,$D0,$1E,$A2,$04,$20
       .byte $68,$FB,$A5,$E1,$09,$80,$85,$E1,$A6,$AD,$A0,$00,$B5,$00,$99,$D5
       .byte $00,$E8,$C8,$C0,$05,$D0,$F5,$A9,$75,$85,$DA,$60,$B5,$00,$38,$F5
       .byte $05,$10,$02,$49,$FF,$24,$E1,$30,$04,$C9,$03,$90,$C4,$C9,$40,$B5
       .byte $00,$B0,$02,$49,$80,$E8,$4C,$F7,$F8,$B5,$00,$38,$F5,$05,$C9,$03
       .byte $90,$17,$C9,$FE,$B0,$13,$A8,$20,$64,$FD,$F0,$0C,$C0,$80,$B4,$00
       .byte $90,$03,$C8,$B0,$01,$88,$94,$00,$60,$20,$64,$FD,$F0,$FA,$A5,$80
       .byte $29,$02,$D0,$F4,$B5,$00,$E8,$4C,$F7,$F8,$B5,$00,$38,$F5,$05,$85
       .byte $A4,$A5,$A0,$4A,$90,$0A,$24,$E1,$50,$06,$B5,$03,$C5,$DE,$B0,$14
       .byte $A5,$A4,$C9,$80,$90,$04,$F6,$00,$B0,$02,$D6,$00,$B5,$00,$24,$E1
       .byte $10,$25,$30,$C5,$B5,$01,$18,$E5,$DC,$10,$02,$49,$FF,$C9,$02,$90
       .byte $06,$C9,$04,$90,$0E,$B0,$B2,$A5,$A4,$C9,$80,$B0,$04,$F6,$00,$90
       .byte $02,$D6,$00,$B5,$00,$49,$80,$E8,$20,$F7,$F8,$A6,$AD,$20,$9D,$F9
       .byte $C9,$46,$B0,$29,$4C,$A0,$FB
LFC85: LDY    COLUP1,X;4
       LDA    LFFD8,Y ;4
       STA    $FC     ;3
       LDA    LFFDE,Y ;4
       STA    $FD     ;3
       JMP.ind ($00FC);5
LFC94: .byte $A5,$80,$29,$0F,$D0,$11,$F6,$06,$B5,$06,$C9,$03,$90,$09,$20,$92
       .byte $FA,$A9,$00,$95,$07,$95,$06,$60,$B5,$00,$18,$69,$20,$95,$00,$B5
       .byte $01,$F6,$06,$B5,$06,$4A,$B0,$EF,$D0,$0A,$B5,$03,$10,$06,$A9,$00
       .byte $95,$07,$95,$06,$20,$9D,$F9,$C9,$1E,$90,$1E,$B5,$05,$C9,$F8,$30
       .byte $0A,$C9,$08,$10,$06,$29,$80,$49,$C0,$D0,$1E,$A5,$80,$C9,$20,$10
       .byte $18,$C9,$E0,$30,$14,$49,$80,$D0,$10,$B5,$05,$49,$87,$85,$AD,$86
       .byte $AC,$E8,$20,$F7,$F8,$A5,$AD,$A6,$AC,$E8,$4C,$F7,$F8,$A0,$20,$B5
       .byte $03,$C9,$1E,$90,$0A,$B5,$06,$69,$06,$95,$06,$10,$02,$A0,$E0,$98
       .byte $18,$75,$05,$85,$AC,$86,$AD,$E8,$20,$F7,$F8,$20,$64,$FD,$F0,$08
       .byte $A6,$AD,$A5,$AC,$E8,$20,$F7,$F8,$A6,$AD,$B5,$03,$C9,$1E,$B0,$2F
       .byte $C9,$14,$B0,$06,$A9,$00,$95,$07,$95,$06,$24,$E1,$30,$21,$B5,$05
       .byte $85,$D5,$A0,$01,$E8,$B5,$00,$99,$D5,$00,$E8,$C8,$C0,$05,$D0,$F5
       .byte $A9,$75,$85,$DA,$A5,$E1,$09,$80,$85,$E1,$A2,$04,$4C,$68,$FB,$60
       .byte $A5,$A0,$C9,$10,$90,$0D,$A5,$80,$4A,$C5,$A0,$B0,$03,$A9,$01,$60
       .byte $A9,$00,$60,$49,$07,$09,$03,$25,$80,$F0,$F2,$D0,$F3,$A5,$BF,$4C
       .byte $6D,$FD
LFD86: STX    $A7     ;3
       STA    $A5     ;3
       LDA    VBLANK,X;4
       SEC            ;2
       SBC.wy $0001,Y ;4
       BPL    LFD97   ;2
       EOR    #$FF    ;2
       SEC            ;2
       ADC    #$00    ;2
LFD97: STA    $A4     ;3
       LDA    RSYNC,X ;4
       SEC            ;2
       SBC.wy $0003,Y ;4
       BPL    LFDA6   ;2
       EOR    #$FF    ;2
       SEC            ;2
       ADC    #$00    ;2
LFDA6: LDX    #$80    ;2
       CMP    $A5     ;3
       BCS    LFDD9   ;2
       LDA    $A4     ;3
       CMP    $A5     ;3
       BCS    LFDD9   ;2
       LDY    $A7     ;3
       LDA.wy $0007,Y ;4
       BEQ    LFDD9   ;2
       LDX    #$00    ;2
       CMP    #$03    ;2
       BEQ    LFDD9   ;2
       CMP    #$04    ;2
       BNE    LFDC6   ;2
       JSR    LFA92   ;6
LFDC6: STX    COLUP0,Y;4
       LDX    COLUP1,Y;4
       LDA    LFFE3,X ;4
       JSR    LF686   ;6
       LDX    #$03    ;2
       STX    COLUP1,Y;4
       JSR    LFB68   ;6
       LDX    #$00    ;2
LFDD9: RTS            ;6

LFDDA: JSR    LF71D   ;6
       CLC            ;2
       ADC    #$02    ;2
       BEQ    LFDE6   ;2
       CMP    #$9F    ;2
       BCC    LFDE8   ;2
LFDE6: LDA    #$01    ;2
LFDE8: STA    $81     ;3
       RTS            ;6

LFDEB: .byte $7A,$9F
LFDED: .byte $C2,$CA
LFDEF: .byte $C6,$69,$B6,$A6,$CE
LFDF4: .byte $1F,$1D
LFDF6: .byte $80,$08,$9E,$10,$00,$F8,$30,$00,$C0,$30,$00,$FE,$FF,$00,$E0,$70
       .byte $01,$FF,$FF,$E0,$F0,$F0,$07,$FF,$FF,$F8,$F8,$A0,$AA,$FF,$80,$01
LFE16: .byte $F9,$F9,$F9,$F9,$F9,$F9,$FA,$FA,$FA,$FA,$FA,$FB,$FB,$FB,$FB,$FB
       .byte $FC,$F5,$F5,$F5,$FC,$F5,$F6,$F6,$F6,$F5,$F6,$F6,$F6,$F8,$EA,$F6
       .byte $F6,$F8,$F8,$EA,$EA,$EA,$F8,$F8,$F5,$EA,$EA,$EA,$EA,$F8,$EA,$EA
       .byte $EA,$EC,$EC,$EA,$EA
LFE4B: .byte $02,$04,$05,$06,$07,$08,$08,$09,$09,$09,$09,$0A,$0A,$0A,$0A,$0A
       .byte $0A,$0A,$0A,$0A,$09,$09,$09,$09,$08,$08,$07,$05,$05,$04,$02,$02
       .byte $03,$04,$05,$06,$06
LFE70: .byte $18,$3C,$7E,$FF,$3C,$FF,$7C,$7E,$FE,$FF,$14,$14,$14,$14,$14,$00
       .byte $14,$14,$14,$10,$10,$10,$10,$12,$12,$14,$14,$12,$12,$10,$10,$12
       .byte $12,$14,$14,$1B,$1B
LFE95: .byte $00,$01,$06,$0B,$10,$15,$1A,$21,$27,$2E,$0D,$0E,$0F,$10,$11,$12
       .byte $13,$01,$01,$02,$03,$04,$05,$06,$07,$07,$08,$08,$09,$09,$09,$09
       .byte $04
LFEB6: .byte $01,$02,$03,$04,$06,$07,$08,$08,$09,$09,$09,$09,$0A,$0A,$0A,$0A
       .byte $0F,$0F,$0F,$0F,$04,$04,$04,$04,$03,$04,$05,$06,$07,$08,$09
LFED5: .byte $09,$74,$71,$68,$5C,$50,$44,$38,$2C
LFEDE: .byte $20,$00,$00,$00,$00,$00,$0B,$05,$0B,$0B
LFEE8: .byte $10,$30,$50,$70,$80,$0A,$1F,$35,$50,$70,$80,$08,$18,$28,$38,$50
       .byte $70,$80
LFEFA: .byte $00,$06,$0C,$12,$19,$1F,$25,$2B,$32,$38,$3E,$44,$4A,$50,$56,$5C
       .byte $62,$68,$6D,$73,$78,$7E,$83,$89,$8E,$93,$98,$9D,$A2,$A7,$AB,$B0
       .byte $B5,$B9,$BD,$C1,$C5,$C9,$CD,$D1,$D4,$D8,$DB,$DE,$E1,$E4,$E7,$EA
       .byte $EC,$EF,$F1,$F3,$F4,$F6,$F8,$F9,$FB,$FC,$FD,$FE,$FE,$FF,$FF,$FF
       .byte $FF
LFF3B: .byte $FC,$FC,$FC,$FC,$F8,$F8,$EE,$F0,$EE,$F0
LFF45: .byte $1B,$0D,$0F,$0F,$07
LFF4A: .byte $05,$02,$06,$06,$02
LFF4F: .byte $0D,$04,$02,$08,$08,$08,$0A,$0A,$0A,$0A,$0B,$0B,$0B,$0C,$0C,$0C
       .byte $0D,$0D,$0D,$0E,$0E,$0E,$0F,$0F,$0F,$0F,$10,$11,$12,$13,$14,$15
       .byte $16,$17,$18,$19,$1A,$1B,$1C,$1D,$1E,$1F,$20,$21,$22,$23,$24,$25
       .byte $26,$27,$29,$2B,$2D,$80,$00,$00,$01,$01,$02,$02,$03,$03,$04,$04
       .byte $05,$05,$06,$06,$07,$07,$08,$08,$09,$09,$0A,$0B,$0B,$0C,$0D,$0D
       .byte $0E,$0F,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$1A,$1B,$1D,$1F
       .byte $25,$30,$40,$54,$80,$03,$09,$12,$16,$1C,$23,$29,$2F,$36,$3D,$43
       .byte $4A,$51,$58,$5F,$66,$6D,$75,$7D,$85,$8D,$95,$9D,$A6,$AF,$B9,$C2
       .byte $CC,$D7,$E2,$EE,$FA,$FF
LFFD5: .byte $04
LFFD6: .byte $0F,$1F
LFFD8: .byte $9F,$7E,$7E,$94,$AC,$01
LFFDE: .byte $FA,$FB,$FB,$FC,$FC
LFFE3: .byte $FD,$10,$30,$00,$50,$20,$50,$20,$3A
LFFEC: STA    LFFF8   ;4
       RTS            ;6

LFFF0: .byte $40,$80
LFFF2: STA    LFFF8   ;4
       JMP    LF019   ;3
LFFF8: .byte $88,$00
LFFFA: .byte $03,$02,$03,$F0
LFFFE: .byte $DB
LFFFF: .byte $D5
