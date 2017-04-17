
import pickle

from extraction.FormatModel.VariableDefinitions import *
from extraction.FormatModel.RawVariableDefinitions import *
import json
def jsonDefault(object):
    return object.__dict__
if __name__ == '__main__':
    Page1 = Category('page1','pagina 1')
    ############
    Localizacion_vivienda = Category('localizacion_de_la_vivienda','Localizacion de la vivienda I')
    #########################
    Ubicacion_geografica = Category('ubicacion_geografica', 'Ubicacion Geografica')


    Departamento_ug = Category('departamento', 'Departamento')
    variable_Departamento_ug = Variable('pos_TL_BR', 'posicion final', None)
    Departamento_ug.addSubType(variable_Departamento_ug)

    Provincia_ug = Category('provincia', 'Provincia')
    variable_Provincia_ug = Variable('pos_TL_BR', 'posicion final', None)
    Provincia_ug.addSubType(variable_Provincia_ug)

    Distrito_ug = Category('distrito', 'Distrito')
    variable_Distrito_ug = Variable('pos_TL_BR', 'posicion final', None)
    Distrito_ug.addSubType(variable_Distrito_ug)

    CentroPoblado_ug = Category('centro_poblado','Centro Poblado')
    variable_CentroPoblado_ug = Variable('pos_TL_BR', 'posicion final', None)
    CentroPoblado_ug.addSubType(variable_CentroPoblado_ug)

    Codigo_cp = Category('codigo_cp', 'Codigo Centro Poblado')
    variable_Codigo_cp = Variable('pos_TL_BR', 'posicion final', None)
    Codigo_cp.addSubType(variable_Codigo_cp)

    NucleoUrbano_ug = Category('nucleo_urbano', 'Nucleo Urbano')
    variable_NucleoUrbano_ug = Variable('pos_TL_BR', 'posicion final', None)
    NucleoUrbano_ug.addSubType(variable_NucleoUrbano_ug)

    Codigo_nu_ug = Category('codigo_nu_ug', 'Codigo Nucleo Urbano')
    variable_Codigo_nu_ug = Variable('pos_TL_BR', 'posicion final', None)
    Codigo_nu_ug.addSubType(variable_Codigo_nu_ug)

    Ubicacion_geografica.addSubType(Departamento_ug)
    Ubicacion_geografica.addSubType(Provincia_ug)
    Ubicacion_geografica.addSubType(Distrito_ug)
    Ubicacion_geografica.addSubType(CentroPoblado_ug)
    Ubicacion_geografica.addSubType(Codigo_cp)
    Ubicacion_geografica.addSubType(NucleoUrbano_ug)
    Ubicacion_geografica.addSubType(Codigo_nu_ug)

    Ubicacion_censal = Category('ubicacion_censal', 'Ubicacion Censal')

    CongN = Category('congN', 'CONG. N')
    variable_CongN = Variable('pos_TL_BR', 'posicion final', None)
    CongN.addSubType(variable_CongN)

    ZonaN = Category('ZonaN', 'Zona. N')
    variable_ZonaN_01 = Variable('pos_TL_BR_01', 'posicion final', None)
    variable_ZonaN_02 = Variable('pos_TL_BR_02', 'posicion final', None)
    ZonaN.addSubType(variable_ZonaN_01)    
    ZonaN.addSubType(variable_ZonaN_02)

    ManzanaN = Category('ManzanaN', 'Manzana Numero')
    variable_ManzanaN_01 = Variable('pos_TL_BR_01', 'posicion final', None)
    variable_ManzanaN_02 = Variable('pos_TL_BR_02', 'posicion final', None)
    ManzanaN.addSubType(variable_ManzanaN_01)
    ManzanaN.addSubType(variable_ManzanaN_02)

    N_Frente = Category('NFrente', 'Numero de frente de manzana')
    variable_N_Frente = Variable('pos_TL_BR', 'posicion final', None)
    N_Frente.addSubType(variable_N_Frente)
    
    ViviendaN = Category('ViviendaN', 'Vivienda numero')
    variable_ViviendaN = Variable('pos_TL_BR', 'posicion final', None)
    ViviendaN.addSubType(variable_ViviendaN)

    NHogares = Category('NHogares', 'Numero de hogares')
    variable_NHogares = Variable('pos_TL_BR', 'posicion final', None)
    NHogares.addSubType(variable_NHogares)

    HogarN = Category('HogarN', 'Hogar Numero')
    variable_HogarN_01 = Variable('pos_TL_BR_01', 'posicion final', None)
    variable_HogarN_02 = Variable('pos_TL_BR_02', 'posicion final', None)
    HogarN.addSubType(variable_HogarN_01)
    HogarN.addSubType(variable_HogarN_02)

    ApNomInformante = Category('informante', 'Apellidos y nombres del informante')
    variable_ApNomInformante = Variable('pos_TL_BR', 'posicion final', None)
    ApNomInformante.addSubType(variable_ApNomInformante)

    NumeroOrden = Category('num_orden', 'Numero de orden')
    variable_NumeroOrden = Variable('pos_TL_BR', 'posicion final', None)
    NumeroOrden.addSubType(variable_NumeroOrden)
    
    NombreVia = Category('nombre_via', 'Nombre de la via')
    variable_NombreVia = Variable('pos_TL_BR', 'posicion final', None)
    NombreVia.addSubType(variable_NombreVia)

    NumeroPuerta = Category('numero_puerta', 'numero de puerta')
    variable_NumeroPuerta_01 = Variable('pos_TL_BR_01', 'posicion final', None)
    variable_NumeroPuerta_02 = Variable('pos_TL_BR_02', 'posicion final', None)
    NumeroPuerta.addSubType(variable_NumeroPuerta_01)
    NumeroPuerta.addSubType(variable_NumeroPuerta_02)

    block = Category('block', 'block')
    variable_block = Variable('pos_TL_BR', 'posicion final', None)
    block.addSubType(variable_block)

    piso = Category('piso', 'piso')
    variable_piso = Variable('pos_TL_BR', 'posicion final', None)
    piso.addSubType(variable_piso)

    interior = Category('interior', 'interior')
    variable_interior = Variable('pos_TL_BR', 'posicion final', None)
    interior.addSubType(variable_interior)

    manzana = Category('manzana', 'manzana')
    variable_manzana = Variable('pos_TL_BR', 'posicion final', None)
    manzana.addSubType(variable_manzana)

    lote = Category('lote', 'lote')
    variable_lote = Variable('pos_TL_BR', 'posicion final', None)
    lote.addSubType(variable_lote)

    km = Category('km', 'km')
    variable_km = Variable('pos_TL_BR', 'posicion final', None)
    km.addSubType(variable_km)

    telefono_domicilio = Category('telefono_domicilio', 'telefono_domicilio')
    variable_telefono_domicilio = Variable('pos_TL_BR', 'posicion final', None)
    telefono_domicilio.addSubType(variable_telefono_domicilio)

    Ubicacion_censal.addSubType(CongN)
    Ubicacion_censal.addSubType(ZonaN)
    Ubicacion_censal.addSubType(ManzanaN)
    Ubicacion_censal.addSubType(N_Frente)
    Ubicacion_censal.addSubType(ViviendaN)
    Ubicacion_censal.addSubType(NHogares)
    Ubicacion_censal.addSubType(HogarN)
    Ubicacion_censal.addSubType(ApNomInformante)
    Ubicacion_censal.addSubType(NumeroOrden)
    Ubicacion_censal.addSubType(NombreVia)
    Ubicacion_censal.addSubType(NumeroPuerta)
    Ubicacion_censal.addSubType(block)
    Ubicacion_censal.addSubType(piso)
    Ubicacion_censal.addSubType(interior)
    Ubicacion_censal.addSubType(manzana)
    Ubicacion_censal.addSubType(lote)
    Ubicacion_censal.addSubType(km)
    Ubicacion_censal.addSubType(telefono_domicilio)

    Localizacion_vivienda.addSubType(Ubicacion_geografica)
    Localizacion_vivienda.addSubType(Ubicacion_censal)

    Page1.addSubType(Localizacion_vivienda)
    #############
    Entrevista_supervision = Category('entrevista_supervision', 'Entrevista y SUpervision II')

    with open('pagina1.json', 'w') as output:
        json.dump(Page1, output, default=jsonDefault, indent=4)

    Page1.describe(True)