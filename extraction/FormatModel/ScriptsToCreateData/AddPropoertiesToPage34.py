from extraction.FormatModel.RawVariableDefinitions import *
from extraction.FormatModel.VariableDefinitions import *

import json


def loadValue(dict_current, cat, cat_grand_parent):
    if dict_current is None:
        return None
    if dict_current['nameParser'] == 'parserImage2ArrayChar':
        if dict_current['nameSingleParser'] == 'letterPredictor':
            print(dict_current['position'])
            return ArrayImageChar(dict_current['position'], dict_current['countItems'])
        if dict_current['nameSingleParser'] == 'digitPredictor':
            return ArrayImageNumber(dict_current['position'], dict_current['countItems'])
    elif dict_current['nameParser'] == 'parserImage2Categoric':
        return loadImageCategoricSimple(dict_current, cat, cat_grand_parent)
    return None


def getLabels(name_cat):
    if name_cat == 'programa_social_beneficiario':
        return [["1", "2", "3", "4", "5", "6", "7"],
                ["8", "9", "10", "11"]]
    elif name_cat == 'presenta_discapacidad':
        return [["1", "2", "3", "4", "5", "6"]]
    elif name_cat == 'sector_desempenho':
        return [["1", "2", "3", "4", "5", "6", "7"],
                ["8", "9", "10"]]
    elif name_cat == 'ultimo_mes_era_un':
        return [["1", "2", "3", "4", "5", "6", "7"],
                ["8", "9", "10"]]
    elif name_cat == 'ultimo_grado_aprobado':
        return [["1", "2", "3", "4", "5", "6"]]
    elif name_cat == 'nivel_educativo':
        return [["1", "2", "3", "4", "5", "6", "7"]]
    elif name_cat == 'sabe_leer_escribir':
        return [["Si", "No"]]
    elif name_cat == 'lengua_materna':
        return [["1", "2", "3", "4", "5", "6", "7"]]
    elif name_cat == 'tipo_seguro':
        return [["1", "2", "3", "4", "5", "6"]]
    elif name_cat == 'estado_civil':
        return [["1", "2", "3", "4", "5", "6"]]
    elif name_cat == 'sexo':
        return [["H", "M", "Si", "No"]]
    elif name_cat == 'num_nucleo_familiar':
        return [["0", "1", "2", "3", "4", "5", "6"]]
    elif name_cat == 'parentesco_jefe_hogar':
        return [["1", "2", "3", "4", "5", "6", "7"],
                ["8", "9", "10", "11"]]
    elif name_cat == 'tipo_vivienda':
        return [["1", "2", "3", "4", "5", "6", "7", "8"]]
    elif name_cat == 'vivienda_es':
        return [["1", "2", "3", "4", "5", "6", "7"]]
    elif name_cat == 'material_predominante_paredes':
        return [["1", "2", "3", "4", "5", "6", "7", "8"]]
    elif name_cat == 'material_predominante_techos':
        return [["1", "2", "3", "4", "5", "6", "7", "8"]]
    elif name_cat == 'material_predominante_pisos':
        return [["1", "2", "3", "4", "5", "6", "7"]]
    elif name_cat == 'alumbrado_vivienda':
        return [["1", "2", "3", "4", "5"]]
    elif name_cat == 'alumbrado_vivienda_no_tiene':
        return [["6"]]
    elif name_cat == 'abastecimiento_agua':
        return [["1", "2", "3", "4", "5", "6", "7"]]
    elif name_cat == 'servicio_higienico':
        return [["1", "2", "3", "4", "5", "6"]]
    elif name_cat == 'mas24_vive_capital':
        return [["1", "2"]]
    elif name_cat == 'combustible_cocinar':
        return [["1", "2", "3", "4", "5", "6", "7"]]
    elif name_cat == 'combustible_cocinar_no_cocina':
        return [["8"]]
    elif name_cat == 'su_hogar_tiene':
        return [["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]]
    elif name_cat == 'tipo_documento':
        return [["1","2"],["3","4"]]
    else:
        return ['']


def loadImageCategoricSimple(dict_current, cat_parent, cat_grand_parent):
    if cat_parent is None:
        return ImageCategoric(dict_current['position'], 1)

    position = dict_current['position']
    position = position[0:2]
    parent_registered = False
    registered_categoric = ['tipo_vivienda', 'vivienda_es', 'material_predominante_paredes', 'material_predominante_techos',
                        'material_predominante_pisos', 'alumbrado_vivienda', 'alumbrado_vivienda_no_tiene',
                        'abastecimiento_agua', 'servicio_higienico', 'mas24_vive_capital', 'combustible_cocinar',
                        'combustible_cocinar_no_cocina', 'su_hogar_tiene','parentesco_jefe_hogar', 'num_nucleo_familiar', 'estado_civil', 'tipo_seguro', 'lengua_materna',
                    'sabe_leer_escribir', 'nivel_educativo', 'ultimo_grado_aprobado', 'ultimo_mes_era_un',
                    'sector_desempenho', 'presenta_discapacidad', 'programa_social_beneficiario','sexo','tipo_documento']


    cat_name = cat_parent.name

    if cat_name in registered_categoric:
        parent_registered = True

    grand_parent_registered = False
    grand_parent_name = ''
    if cat_grand_parent is not None and cat_grand_parent.name in registered_categoric:
        grand_parent_registered = True
        grand_parent_name = cat_grand_parent.name

    if parent_registered:
        position.append(getLabels(cat_name))
    elif grand_parent_registered:
        position.append(getLabels(grand_parent_name))



    if cat_name in ['parentesco_jefe_hogar', 'num_nucleo_familiar', 'estado_civil', 'tipo_seguro', 'lengua_materna',
                    'sabe_leer_escribir', 'nivel_educativo', 'ultimo_grado_aprobado', 'ultimo_mes_era_un',
                    'sector_desempenho', 'presenta_discapacidad', 'programa_social_beneficiario']:
        return ImageCategoricLabelsInside(position, 1)
    else:
        if cat_name in ['sexo']:
            return ImageCategoricLabelsSex(position,1)
        else:
            if cat_name in ['tipo_documento']:
                return ImageCategoricLabelsDocumento(position,1)
            else:
                if parent_registered or grand_parent_registered:

                    return ImageCategoricLabelsLeft(position, 1)

                print(cat_name)
                return ImageCategoric(dict_current['position'], 1)


def loadCategory(dict_current, cat_parent=None, cat_grand_parent = None):
    cat = Category(dict_current['name'], dict_current['description'])
    cat.value = loadValue(dict_current['value'], cat_parent, cat_grand_parent)
    cat.hasValue = dict_current['hasValue']
    for sub in dict_current['subTypes']:
        r = loadCategory(sub, cat, cat_parent)
        cat.addSubType(r)
    return cat


def jsonDefault(object):
    return object.__dict__


if __name__ == '__main__':
    str_number = '4'
    str_file = '../pagina' + str_number + '.json'
    str_file_output = '../pagina' + str_number + '_test.json'
    with open(str_file, 'r') as input:
        dict_Page = json.load(input)
        Page = loadCategory(dict_Page)
        print(Page)

    with open(str_file_output, 'w') as output:
        json.dump(Page, output, default=jsonDefault, indent=4)
