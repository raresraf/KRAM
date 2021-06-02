import axios from 'axios'
import { GoogleResponse } from './types'

export async function getAnswer(question: string): Promise<string> {
  const response = await axios.post('http://localhost:5000/answer', {
    question
  })

  return response.data.answer as string
}

export async function getGoogleDescription(answer: string): Promise<GoogleResponse> {
  const encodedAnswer = encodeURI(answer)
  const requestUrl = `https://kgsearch.googleapis.com/v1/entities:search?query=${encodedAnswer}&key=AIzaSyDO4qwDausPvQZ-cAgyNkduChjsLcbz4WE&limit=5&indent=True`

  const response = await axios.get(requestUrl)

  return response.data as GoogleResponse
}