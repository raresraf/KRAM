export interface GoogleResponse {
  itemListElement?: (ItemListElementEntity)[] | null
}

export interface ItemListElementEntity {
  result: Result
  resultScore: number
}

export interface Result {
  detailedDescription: DetailedDescription
  description: string
  name: string
}

export interface DetailedDescription {
  url: string
  articleBody: string
  license: string
}
