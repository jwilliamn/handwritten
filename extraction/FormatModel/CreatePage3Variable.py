import pickle

from extraction.FormatModel.VariableDefinitions import *
from extraction.FormatModel.RawVariableDefinitions import *
import json


def jsonDefault(object):
    return object.__dict__


if __name__ == '__main__':
    Page3 = Category('page3', 'pagina 3')
    ############
    for r in range(1,6):
        str_r = str(r)
        if len(str_r) == 1:
            str_r = '0'+str_r
        P = Category('P'+str_r,'Persona '+str_r)
        
        ap_paterno=Category('apellido_paterno','Apellido paterno')
        variable_ap_paterno=Variable('pos_TL_BR','posicion final', None)
        ap_paterno.addSubType(variable_ap_paterno)
    
        ap_materno = Category('apellido_materno', 'Apellido materno')
        variable_ap_materno = Variable('pos_TL_BR', 'posicion final', None)
        ap_materno.addSubType(variable_ap_materno)
    
        nombres = Category('nombres', 'nombres')
        variable_nombres = Variable('pos_TL_BR', 'posicion final', None)
        nombres.addSubType(variable_nombres)
    
        fecha_nacimiento = Category('fecha_nacimiento', 'Fecha de nacimiento')
        variable_fecha_nacimiento = Variable('pos_TL_BR', 'posicion final', None)
        fecha_nacimiento.addSubType(variable_fecha_nacimiento)
    
        edad_anhos = Category('edad_anhos', 'edad_anios')
        variable_edad_anhos = Variable('pos_TL_BR', 'posicion final', None)
        edad_anhos.addSubType(variable_edad_anhos)
    
        edad_meses = Category('edad_meses', 'edad_meses')
        variable_edad_meses = Variable('pos_TL_BR', 'posicion final', None)
        edad_meses.addSubType(variable_edad_meses)
        
        tipo_documento = Category('tipo_documento', 'tipo_documento')
        variable_tipo_documento = Variable('pos_TL_BR', 'posicion final', None)
        tipo_documento.addSubType(variable_tipo_documento)
        
        
        numero_documento = Category('numero_documento', 'numero_documento')
        variable_numero_documento = Variable('pos_TL_BR', 'posicion final', None)
        numero_documento.addSubType(variable_numero_documento)
        
        parentesco_jefe_hogar = Category('parentesco_jefe_hogar', 'parentesco_jefe_hogar')
        variable_parentesco_jefe_hogar = Variable('pos_TL_BR', 'posicion final', None)
        parentesco_jefe_hogar.addSubType(variable_parentesco_jefe_hogar)
        
        num_nucleo_familiar = Category('num_nucleo_familiar', 'num_nucleo_familiar')
        variable_num_nucleo_familiar = Variable('pos_TL_BR', 'posicion final', None)
        num_nucleo_familiar.addSubType(variable_num_nucleo_familiar)
    
        sexo = Category('sexo', 'sexo')
        variable_sexo = Variable('pos_TL_BR', 'posicion final', None)
        sexo.addSubType(variable_sexo)
        
        estado_civil = Category('estado_civil', 'estado_civil')
        variable_estado_civil = Variable('pos_TL_BR', 'posicion final', None)
        estado_civil.addSubType(variable_estado_civil)
        
        tipo_seguro = Category('tipo_seguro', 'tipo_seguro')
        variable_tipo_seguro = Variable('pos_TL_BR', 'posicion final', None)
        tipo_seguro.addSubType(variable_tipo_seguro)
    
        lengua_materna = Category('lengua_materna', 'lengua_materna')
        variable_lengua_materna = Variable('pos_TL_BR', 'posicion final', None)
        lengua_materna.addSubType(variable_lengua_materna)
        
        sabe_leer_escribir = Category('sabe_leer_escribir', 'sabe_leer_escribir')
        variable_sabe_leer_escribir = Variable('pos_TL_BR', 'posicion final', None)
        sabe_leer_escribir.addSubType(variable_sabe_leer_escribir)
        
        nivel_educativo = Category('nivel_educativo', 'nivel_educativo')
        variable_nivel_educativo = Variable('pos_TL_BR', 'posicion final', None)
        nivel_educativo.addSubType(variable_nivel_educativo)
        
        ultimo_grado_aprobado = Category('ultimo_grado_aprobado', 'ultimo_grado_aprobado')
        variable_ultimo_grado_aprobado = Variable('pos_TL_BR', 'posicion final', None)
        ultimo_grado_aprobado.addSubType(variable_ultimo_grado_aprobado)
        
        ultimo_mes_era_un = Category('ultimo_mes_era_un', 'ultimo_mes_era_un')
        variable_ultimo_mes_era_un = Variable('pos_TL_BR', 'posicion final', None)
        ultimo_mes_era_un.addSubType(variable_ultimo_mes_era_un)
    
        sector_desempenho = Category('sector_desempenho', 'sector_desempenho')
        variable_sector_desempenho = Variable('pos_TL_BR', 'posicion final', None)
        sector_desempenho.addSubType(variable_sector_desempenho)
    
        presenta_discapacidad = Category('presenta_discapacidad', 'presenta_discapacidad')
        variable_presenta_discapacidad = Variable('pos_TL_BR', 'posicion final', None)
        presenta_discapacidad.addSubType(variable_presenta_discapacidad)
    
        programa_social_beneficiario = Category('programa_social_beneficiario', 'programa_social_beneficiario')
        variable_programa_social_beneficiario = Variable('pos_TL_BR', 'posicion final', None)
        programa_social_beneficiario.addSubType(variable_programa_social_beneficiario)
        #############
    
        P.addSubType(ap_paterno)
        P.addSubType(ap_materno)
        P.addSubType(nombres)
        P.addSubType(fecha_nacimiento)
        P.addSubType(edad_anhos)
        P.addSubType(edad_meses)
        P.addSubType(tipo_documento)
        P.addSubType(numero_documento)
        P.addSubType(parentesco_jefe_hogar)
        P.addSubType(num_nucleo_familiar)
        P.addSubType(sexo)
        P.addSubType(estado_civil)
        P.addSubType(tipo_seguro)
        P.addSubType(lengua_materna)
        P.addSubType(sabe_leer_escribir)
        P.addSubType(nivel_educativo)
        P.addSubType(ultimo_grado_aprobado)
        P.addSubType(ultimo_mes_era_un)
        P.addSubType(sector_desempenho)
        P.addSubType(presenta_discapacidad)
        P.addSubType(programa_social_beneficiario)
    
    
        Page3.addSubType(P)

    with open('pagina3.json', 'w') as output:
        json.dump(Page3, output, default=jsonDefault, indent=4)

    Page3.describe(True)