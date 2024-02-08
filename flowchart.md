flowchart TD
    subgraph id_pre [Pre-processing]
    direction LR
    id_pre_3[Format trade matrix and production vector] --> id_pre_4[Rename country, item labels]
    end
    subgraph id_mod [Model]
    direction LR
    id_mod_2[Remove ''net zero'' countries] --> id_mod_3[Pre-balance trade matrix]
    id_mod_3 --> id_mod_4[Correct for re-exports]
    id_mod_4 --> id_mod_5[Replace trade matrix diagonal with zeroes]
    id_mod_5 -->  id_mod_6[Filter countries]
    id_mod_6 --> id_mod_7[Remove countries below certain trade volume]
    id_mod_7 --> id_mod_8[Apply scenario]
    id_mod_8 --> id_mod_9[Apply gravity model of trade]
    id_mod_9 --> id_mod_10[Build graph]
    id_mod_10 --> id_mod_11[Find communities]
    end
    subgraph id_pos [Post-processing]
    direction LR
    end
    FAO[(FAO data)] ==> id_pre
    id_pre ==> id_mod
    id_mod ==> id_pos
    id_pos ==o RES((Results))