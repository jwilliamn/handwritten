import pickle

from extraction.FormatModel.VariableDefinitions import *
from extraction.FormatModel.RawVariableDefinitions import *
import json


def jsonDefault(object):
    return object.__dict__


if __name__ == '__main__':
    Page2 = Category('page2', 'pagina 2')
    ############
    Caracteristicas_vivienda = Category('caracteristicas_vivienda', 'Caracteristicas de la vivienda')
    #########################
    tipo_vivienda = Category('tipo_vivienda', 'Tipo de Vivienda')
    
    tipo_vivienda_num = Category('num', 'Numero correspondiente al tipo de vivienda')
    variable_tipo_vivienda_num = Variable('pos_TL_BR', 'posicion final', None)
    tipo_vivienda_num.addSubType(variable_tipo_vivienda_num)

    tipo_vivienda_otro = Category('tipo_vivienda_otro', 'otr Tipo de Vivienda')
    variable_tipo_vivienda_otro = Variable('pos_TL_BR', 'posicion final', None)
    tipo_vivienda_otro.addSubType(variable_tipo_vivienda_otro)
    
    tipo_vivienda.addSubType(tipo_vivienda_num)
    tipo_vivienda.addSubType(tipo_vivienda_otro)


    vivienda_es = Category('vivienda_es', 'Su vivienda es')

    vivienda_es_num = Category('num', 'Numero correspondiente al su vivienda es')
    variable_vivienda_es_num = Variable('pos_TL_BR', 'posicion final', None)
    vivienda_es_num.addSubType(variable_vivienda_es_num)

    vivienda_es_otro = Category('vivienda_es_otro', 'otro Su vivienda es')
    variable_vivienda_es_otro = Variable('pos_TL_BR', 'posicion final', None)
    vivienda_es_otro.addSubType(variable_vivienda_es_otro)

    vivienda_es.addSubType(vivienda_es_num)
    vivienda_es.addSubType(vivienda_es_otro)



    material_predominante_paredes = Category('material_predominante_paredes',
                                             'Material predominante en las paredes exteriores')

    material_predominante_paredes_num = Category('num',
                                                 'Numero correspondiente al material predominante en paredes exteriores')
    variable_material_predominante_paredes_num = Variable('pos_TL_BR', 'posicion final', None)
    material_predominante_paredes_num.addSubType(variable_material_predominante_paredes_num)

    material_predominante_paredes_otro = Category('material_predominante_paredes_otro',
                                                  'Otro material predominante en paredes exteriores')
    variable_material_predominante_paredes_otro = Variable('pos_TL_BR', 'posicion final', None)
    material_predominante_paredes_otro.addSubType(variable_material_predominante_paredes_otro)

    material_predominante_paredes.addSubType(material_predominante_paredes_num)
    material_predominante_paredes.addSubType(material_predominante_paredes_otro)


    material_predominante_techos = Category('material_predominante_techos', 'Material predominante en los techos')

    material_predominante_techos_num = Category('num', 'Numero correspondiente al Material predominante en los techos')
    variable_material_predominante_techos_num = Variable('pos_TL_BR', 'posicion final', None)
    material_predominante_techos_num.addSubType(variable_material_predominante_techos_num)

    material_predominante_techos_otro = Category('material_predominante_techos_otro',
                                                 'otro Material predominante en los techos')
    variable_material_predominante_techos_otro = Variable('pos_TL_BR', 'posicion final', None)
    material_predominante_techos_otro.addSubType(variable_material_predominante_techos_otro)

    material_predominante_techos.addSubType(material_predominante_techos_num)
    material_predominante_techos.addSubType(material_predominante_techos_otro)

    material_predominante_pisos = Category('material_predominante_pisos', 'Material predominante en los pisos')

    material_predominante_pisos_num = Category('num', 'Numero correspondiente al Material predominante en los pisos')
    variable_material_predominante_pisos_num = Variable('pos_TL_BR', 'posicion final', None)
    material_predominante_pisos_num.addSubType(variable_material_predominante_pisos_num)

    material_predominante_pisos_otro = Category('material_predominante_pisos_otro',
                                                 'otro Material predominante en los pisos')
    variable_material_predominante_pisos_otro = Variable('pos_TL_BR', 'posicion final', None)
    material_predominante_pisos_otro.addSubType(variable_material_predominante_pisos_otro)

    material_predominante_pisos.addSubType(material_predominante_pisos_num)
    material_predominante_pisos.addSubType(material_predominante_pisos_otro)

    alumbrado_vivienda = Category('alumbrado_vivienda', 'Alumbrado que tiene la vivienda')

    alumbrado_vivienda_num = Category('num', 'Numero correspondiente al Alumbrado que tiene la vivienda')
    variable_alumbrado_vivienda_num = Variable('pos_TL_BR', 'posicion final', None)
    alumbrado_vivienda_num.addSubType(variable_alumbrado_vivienda_num)

    alumbrado_vivienda_otro = Category('alumbrado_vivienda_otro', 'otro Alumbrado que tiene la vivienda')
    variable_alumbrado_vivienda_otro = Variable('pos_TL_BR', 'posicion final', None)
    alumbrado_vivienda_otro.addSubType(variable_alumbrado_vivienda_otro)

    alumbrado_vivienda_no_tiene = Category('alumbrado_vivienda_no_tiene', 'no tiene alumbrado en la vivienda')
    variable_alumbrado_vivienda_no_tiene = Variable('pos_TL_BR', 'posicion final', None)
    alumbrado_vivienda_no_tiene.addSubType(variable_alumbrado_vivienda_no_tiene)

    alumbrado_vivienda.addSubType(alumbrado_vivienda_num)
    alumbrado_vivienda.addSubType(alumbrado_vivienda_otro)
    alumbrado_vivienda.addSubType(alumbrado_vivienda_no_tiene)

    abastecimiento_agua = Category('abastecimiento_agua', 'El abastecimiento de agua procede de')

    abastecimiento_agua_num = Category('num', 'Numero correspondiente a El abastecimiento de agua procede de')
    variable_abastecimiento_agua_num = Variable('pos_TL_BR', 'posicion final', None)
    abastecimiento_agua_num.addSubType(variable_abastecimiento_agua_num)

    abastecimiento_agua_otro = Category('abastecimiento_agua_otro',
                                                'otro El abastecimiento de agua procede de')
    variable_abastecimiento_agua_otro = Variable('pos_TL_BR', 'posicion final', None)
    abastecimiento_agua_otro.addSubType(variable_abastecimiento_agua_otro)

    abastecimiento_agua.addSubType(abastecimiento_agua_num)
    abastecimiento_agua.addSubType(abastecimiento_agua_otro)

    servicio_higienico = Category('servicio_higienico', 'El servicio higienico conectado a')

    servicio_higienico_num = Category('num', 'Numero correspondiente a El servicio higienico conectado a')
    variable_servicio_higienico_num = Variable('pos_TL_BR', 'posicion final', None)
    servicio_higienico_num.addSubType(variable_servicio_higienico_num)

    servicio_higienico.addSubType(servicio_higienico_num)

    horas_a_capital_distrital = Category('horas_a_capital_distrital',
                                         'Cuantas horas demoran en llegar desde su vivienda a la capital distrital')

    horas_a_capital_distrital_num = Category('horas', 'Horas')
    variable_horas_a_capital_distrital_num = Variable('pos_TL_BR', 'posicion final', None)
    horas_a_capital_distrital_num.addSubType(variable_horas_a_capital_distrital_num)

    horas_a_capital_0_24 = Category('mas24_vive_capital',
                                    'Opciones 1 o 2, mas de 24 horas o vive en la capital distrital')
    variable_horas_a_capital_0_24 = Variable('pos_TL_BR', 'posicion final', None)
    horas_a_capital_0_24.addSubType(variable_horas_a_capital_0_24)

    horas_a_capital_distrital.addSubType(horas_a_capital_distrital_num)
    horas_a_capital_distrital.addSubType(horas_a_capital_0_24)

    Caracteristicas_vivienda.addSubType(tipo_vivienda)
    Caracteristicas_vivienda.addSubType(vivienda_es)
    Caracteristicas_vivienda.addSubType(material_predominante_paredes)
    Caracteristicas_vivienda.addSubType(material_predominante_techos)
    Caracteristicas_vivienda.addSubType(material_predominante_pisos)
    Caracteristicas_vivienda.addSubType(alumbrado_vivienda)
    Caracteristicas_vivienda.addSubType(abastecimiento_agua)
    Caracteristicas_vivienda.addSubType(servicio_higienico)
    Caracteristicas_vivienda.addSubType(horas_a_capital_distrital)

    #############

    Datos_del_hogar = Category('datos_del_hogar', 'Datos del hogar')
    
    num_habitaciones = Category('num_habitaciones','Numero de habitaciones')
    variable_num_habitaciones = Variable('pos_TL_BR','posicion final', None)
    num_habitaciones.addSubType(variable_num_habitaciones)

    combustible_cocinar = Category('combustible_cocinar', 'Combustible que se usa en el hogar para cocinar')

    combustible_cocinar_num = Category('num',
                                       'Numero correspondiente al Combustible que se usa en el hogar para cocinar')
    variable_combustible_cocinar_num = Variable('pos_TL_BR', 'posicion final', None)
    combustible_cocinar_num.addSubType(variable_combustible_cocinar_num)

    combustible_cocinar_otro = Category('combustible_cocinar_otro',
                                        'otro Combustible que se usa en el hogar para cocinar')
    variable_combustible_cocinar_otro = Variable('pos_TL_BR', 'posicion final', None)
    combustible_cocinar_otro.addSubType(variable_combustible_cocinar_otro)

    combustible_cocinar_no_cocina = Category('combustible_cocinar_no_cocina',
                                            'no cocina')
    variable_combustible_cocinar_no_tiene = Variable('pos_TL_BR', 'posicion final', None)
    combustible_cocinar_no_cocina.addSubType(variable_combustible_cocinar_no_tiene)

    combustible_cocinar.addSubType(combustible_cocinar_num)
    combustible_cocinar.addSubType(combustible_cocinar_otro)
    combustible_cocinar.addSubType(combustible_cocinar_no_cocina)

    su_hogar_tiene = Category('su_hogar_tiene', 'El hogar tiene (artefactos)')

    su_hogar_tiene_num = Category('num', 'Numero correspondiente a El hogar tiene')
    variable_su_hogar_tiene_num = Variable('pos_TL_BR', 'posicion final', None)
    su_hogar_tiene_num.addSubType(variable_su_hogar_tiene_num)

    su_hogar_tiene.addSubType(su_hogar_tiene_num)

    num_suministro = Category('numero_suministro','Numero de suministro, de luz, agua o no tiene')
    tipo_suministro = Category('tipo_suministro','Tipo de suministro')
    variable_tipo_suministro = Variable('pos_TL_BR', 'posicion final', None)
    tipo_suministro.addSubType(variable_tipo_suministro)
    num_suministro_numero = Category('numero_suministro', 'Numero de suministro')
    variable_num_suministro_numero = Variable('pos_TL_BR', 'posicion final', None)
    num_suministro_numero.addSubType(variable_num_suministro_numero)

    num_suministro.addSubType(variable_tipo_suministro)
    num_suministro.addSubType(num_suministro_numero)

    personas_permanentes_hogar = Category('personas_hogar','Cantidad de personas que viven permanentemente en el hogar')
    
    total_personas = Category('total','Total')
    variable_total_personas = Variable('pos_TL_BR', 'posicion final', None)
    total_personas.addSubType(variable_total_personas)

    total_hombres = Category('cantidad_hombres', 'Hombres')
    variable_total_hombres = Variable('pos_TL_BR', 'posicion final', None)
    total_hombres.addSubType(variable_total_hombres)

    total_mujeres = Category('cantidad_mujeres', 'Mujeres')
    variable_total_mujeres = Variable('pos_TL_BR', 'posicion final', None)
    total_mujeres.addSubType(variable_total_mujeres)

    personas_permanentes_hogar.addSubType(total_personas)
    personas_permanentes_hogar.addSubType(total_hombres)
    personas_permanentes_hogar.addSubType(total_mujeres)

    Datos_del_hogar.addSubType(num_habitaciones)
    Datos_del_hogar.addSubType(combustible_cocinar)
    Datos_del_hogar.addSubType(su_hogar_tiene)
    Datos_del_hogar.addSubType(num_suministro)
    Datos_del_hogar.addSubType(personas_permanentes_hogar)

    Page2.addSubType(Caracteristicas_vivienda)
    Page2.addSubType(Datos_del_hogar)
    with open('pagina2.json', 'w') as output:
        json.dump(Page2, output, default=jsonDefault, indent=4)

    Page2.describe(True)