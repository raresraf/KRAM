import './index.scss'

import * as React from 'react'

import ResultsList from '../ResultsList'

import Logo from './assets/Logo.svg'

import {getAnswer, getGoogleDescription} from '../../services/'

export default function Main() {
  const [question, setQuestion] = React.useState('')
  const [answer, setAnswer] = React.useState('')
  const [answerDescription, setAnswerDescription] = React.useState('')
  const [isLoading, setIsLoading] = React.useState(false)
  const [isShowingResults, setIsShowingResults] = React.useState(false)

  const onTextInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuestion(e.target.value)
  }

  const onEnterKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      setAnswerDescription('')
      onSearch()
    }
  }

  const onSearch = async () => {
    setIsShowingResults(true)
    setIsLoading(true)

    try {
      const answer = await getAnswer(question)
      const descriptionObject = await getGoogleDescription(answer)
  
      if (descriptionObject.itemListElement && descriptionObject.itemListElement.length > 0) {
        setAnswerDescription(descriptionObject.itemListElement[0].result.detailedDescription.articleBody)
      }
  
      setAnswer(answer)
    } catch(e) {
      setAnswer('Not Found')
    }
    setIsLoading(false)
  }

  return (
    <div className='Main'>
      <img className='Logo' src={Logo} alt='' />
      <input onChange={onTextInputChange} placeholder='Ask me a question about movies' onKeyDown={onEnterKeyDown}/>
      <button onClick={onSearch}>KRAM Search</button>
      <ResultsList
        response={answer}
        description={answerDescription}
        isLoading={isLoading}
        isExpanded={isShowingResults}
      />
    </div>
  )
}
